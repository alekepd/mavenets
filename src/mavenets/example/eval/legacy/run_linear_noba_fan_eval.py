"""Evaluate a linear model using a custom cross-dataset setup.

The model is first trained on all training sets using per-experiment heads.
The per-experiment heads are then extracted and the parameters are frozen. A
new linear is associated with the frozen head and trained on a subset of the
original dataset.

This is intended to first infer the per-experiment relationships and then
test cross-experiment extrapolation in a particular way.

This script does not search through hyperparameters.
"""
from typing import Final, List
from itertools import product
import torch
import pandas as pd  # type: ignore
from ....data import get_datasets, CORE_DATA_SPECS
from ....network import MLP, SharedFanTuner
from ....tools import train_tunable_model
from ....report import predict

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 2


def test_linear(
    n_epochs: int,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    grad_clip: int = 300,
    fan_size: int = 16,
    pretrain_n_epochs: int = 1000,
) -> Tuple:
    """Train model and evaluate, but with many quirks.

    The model is trained with experimental heads on all datasets; then, the experimental
    heads are frozen and the rest of the model is reset. The model is then subsequently
    retrained holding the experimental heads frozen.

    A fan head is used.

    The first training portion uses data from all experiments. The second part omits
    BA1 and BA2 from the training set and uses BA1 as a validation set.

    This function returns a model and a dataframe with the test predictions.
    However, the test predictions are obtained from the model at the end of
    training, which may be overtrained. Early stopping is performed, but the model
    parameters are not rolled back to the optimal epoch after stopping training.

    The arguments control the network architecture and training process.
    """

    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=[],
        pre_flatten=True,
        post_squeeze=True,
    )
    model = SharedFanTuner(underlying_model, n_heads=8, fan_size=fan_size).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    results = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=pretrain_n_epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        report_datasets=report_datasets,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    non_ba_specs = [x for x in CORE_DATA_SPECS if x.name not in ["BA1","BA2"]]

    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=non_ba_specs,
        val_specs=non_ba_specs,
        feat_type="onehot",
    )

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=[],
        pre_flatten=True,
        post_squeeze=True,
    ).to(DEVICE)
    model.base_model = underlying_model
    opter = torch.optim.AdamW(
        underlying_model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    results = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=n_epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        report_datasets=report_datasets,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        compile_mode="max-autotune",
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        start_loss_param=0.0,
        progress_bar=False,
    )

    # get test data
    _, _, eval_data = get_datasets(device=DEVICE, feat_type="onehot", include_test=True)
    pred_table = predict(model=model, dataset=eval_data, batch_size=1024)

    return model, pred_table


def run() -> None:
    """Train a model using a given set of hypers n times and evaluate the best.

    Note that n_epochs is here the maximum number of epochs attempted; each run
    uses early stopping to determine the optimal number of epochs to use for
    the saved model.
    """
    fan_size = 4
    wdecay = 0.001
    n_epochs = 90
    lr = 1e-4
    model, table = test_linear(
        weight_decay=wdecay,
        n_epochs=n_epochs,
        learning_rate=lr,
        fan_size=fan_size,
    )
    core_name = "TESTEVAL_linear_wdecay{}_learningrate{}_fan{}_batest_nepoch{}".format(wdecay, lr, fan_size, n_epochs)
    table.to_csv(core_name+".csv")
    torch.save(model, core_name+".pt")


if __name__ == "__main__":
    scan()

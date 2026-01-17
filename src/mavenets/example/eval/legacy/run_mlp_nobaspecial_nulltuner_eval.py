"""Evaluate single MLP architecture on test set.

Model is trained on all data except for the BA1 and BA2 sets. No per-experiment heads
are used.
"""
from typing import Final, List
from itertools import product
import torch
import pandas as pd  # type: ignore
from ....data import get_datasets, CORE_DATA_SPECS
from ....network import MLP, NullTuner
from ....tools import train_tunable_model
from ....report import predict

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 2


def test_mlp(
    hidden_layer_sizes: List[int],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
) -> Tuple:
    """Train model and evaluate, but with quirks.

    MLP is trained on all but BA1 and BA2; early stopping is done using BA1.

    This function returns a model and a dataframe with the test predictions.
    However, the test predictions are obtained from the model at the end of
    training, which may be overtrained. Early stopping is performed, but the model
    parameters are not rolled back to the optimal epoch after stopping training.

    The arguments control the network architecture and training process.
    """
    non_ba_specs = [x for x in CORE_DATA_SPECS if x.name not in ["BA1","BA2"]]

    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=non_ba_specs,
        val_specs=non_ba_specs,
        feat_type="onehot",
    )

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
        pre_flatten=True,
        post_squeeze=True,
    )
    model = NullTuner(underlying_model).to(DEVICE)
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
        n_epochs=n_epochs,
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

    # get test data
    _, _, eval_data = get_datasets(device=DEVICE, feat_type="onehot", include_test=True)
    pred_table = predict(model=model, dataset=eval_data, batch_size=1024)

    return model, pred_table


def run() -> None:
    """Evaluate performance of a given set of hypers."""
    layer_sel =  [256, 8]
    wdecay = 0.001
    n_epochs = 25 # this is used imperfectly for early stopping in test_mlp.
    lr = 1e-4
    model, table = test_mlp(
        hidden_layer_sizes=layer_sel,
        weight_decay=wdecay,
        n_epochs=n_epochs,
        learning_rate=lr,
    )
    core_name = "TESTEVAL_mlp_l{}_wdecay{}_learningrate{}_noba-special_nepoch{}".format(repr(layer_sel), wdecay, lr, n_epochs)
    table.to_csv(core_name+".csv")
    torch.save(model, core_name+".pt")


if __name__ == "__main__":
    run()

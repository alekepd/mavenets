"""Evaluate a linear model using a custom cross-dataset setup.

The model is first trained on all training sets using per-experiment heads.
The per-experiment heads are then extracted and the parameters are frozen. A
new linear is associated with the frozen head and trained on a subset of the
original dataset.

This is intended to first infer the per-experiment relationships and then
test cross-experiment extrapolation in a particular way.

This script does not search through hyperparameters.
"""
from typing import Final, List, Tuple
from itertools import product
import torch
import pandas as pd  # type: ignore
from ...data import get_datasets, CORE_DATA_SPECS
from ...network import MLP, SharedFanTuner
from ...tools import train_tunable_model
from ...report import predict

# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 2


def test_linear(
    fan_size: int,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
) -> Tuple:
    """Train model and evaluate, but with many quirks.

    The model is trained with experimental heads on all datasets; then, the experimental
    heads are frozen and the rest of the model is reset. The model is then subsequently
    retrained holding the experimental heads frozen.

    A fan head is used.

    The first training portion uses data from all experiments. The second part omits
    BA1 and BA2 from the training set and uses BA1 as a validation set.

    This function returns the best_epoch, the best validation score, a model,
    a dataframe describing training, and a dataframe with the test predictions.
    However, the test predictions are obtained from the model at the end of
    training, not that of the returned epoch index. This function primarily makes
    sense in the context of the "helper" function.

    The arguments control the network architecture and training process.
    """

    # collate all datasets
    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    # create linear model
    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=[],
        pre_flatten=True,
        post_squeeze=True,
    )
    # create per experiment heads
    model = SharedFanTuner(underlying_model, n_heads=8, fan_size=fan_size).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    # train model with heads. We train for 500 epochs here just to make sure the
    # experiment heads are somewhat trained, but this is a pretty arbitrary value.
    # This is higher than in some MLP runs because the linear models don't overfit.
    best_epoch, best_val, training_record = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=500,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        report_datasets=report_datasets,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        compile_mode="max-autotune",
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    # train is now all datasets except for BA1 and BA2.
    non_ba_specs = [x for x in CORE_DATA_SPECS if x.name not in ["BA1","BA2"]]

    # we use BA1 as the val set.
    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=non_ba_specs,
        val_specs=['BA1'],
        feat_type="onehot",
    )

    # make new linear model to associate with frozen heads
    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=[],
        pre_flatten=True,
        post_squeeze=True,
    ).to(DEVICE)
    # associate new model with heads
    model.base_model = underlying_model
    # define optimizer to only use the non-head parameters
    opter = torch.optim.AdamW(
        underlying_model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    # train model (holding the per experiment heads constant)
    best_epoch, best_val, training_record = train_tunable_model(
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

    return best_epoch, best_val, model, training_record, pred_table


def helper(seed: int, **kwargs):
    """Evaluate a model.

    First determines the optimal number of training epochs and then retrains.

    Returns the model, the validation score, and the predictions on the test set.
    """
    torch.manual_seed(seed)
    best_epoch, _, _, _, _ = test_linear(
            **kwargs
    )

    new_kwargs = kwargs.copy()
    new_kwargs["n_epochs"] = best_epoch

    torch.manual_seed(seed)
    _, _, model, training_report, test_pred = test_linear(
            **new_kwargs
    )

    valid = training_report.valid.iloc[-1]
    return (model, valid, test_pred)


def run(attempt_seeds: Tuple= (1234231, 54636, 2931243)) -> None:
    """Train a model using a given set of hypers n times and evaluate the best.

    Note that n_epochs is here the maximum number of epochs attempted; each run
    uses early stopping to determine the optimal number of epochs to use for
    the saved model.
    """
    fan_size = 4
    wdecay = 0.001
    n_epochs = 100 # this the max number of epochs considered; early stopping is used.
    lr = 1e-4
    record = {}
    for seed in attempt_seeds:
        model, valid, test_pred = helper(
            seed=seed,
            fan_size=fan_size,
            weight_decay=wdecay,
            n_epochs=n_epochs,
            learning_rate=lr,
        )
        record.update({valid: (model, test_pred)})
    best_model, best_table = record[min(record.keys())]
    print("val from random inits:", list(record.keys()))
    core_name = "TESTEVAL_linear_wdecay{}_learningrate{}_batest_fan{}_multiattempt".format(wdecay, lr, fan_size)
    best_table.to_csv(core_name+".csv")
    torch.save(best_model, core_name+".pt")


if __name__ == "__main__":
    run()

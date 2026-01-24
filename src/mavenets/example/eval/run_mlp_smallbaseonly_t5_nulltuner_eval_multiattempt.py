"""Evaluate a fixed MLP architecture on test sets while only training on base data.

No per-experiment heads are used.
"""
from typing import Final, List, Tuple
import torch
from torch.utils.data import Dataset
from ...data import get_datasets
from ...network import MLP, NullTuner
from ...tools import train_tunable_model
from ...report import predict

# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 2


def test_mlp(
    eval_dataset: Dataset,
    hidden_layer_sizes: List[int],
    train_dataset: Dataset,
    valid_dataset: Dataset,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
    dropout: float = 0.0,
) -> Tuple:
    """Train model and evaluate on the test set, with quirks.

    A MLP is trained on data from the base experiment. No experimental head is used.

    This function returns the best_epoch, the best  validation score, a model,
    a dataframe describing training, and a dataframe with the test predictions.
    However, the test predictions are obtained from the model at the end of
    training, not that of the returned epoch index. This function primarily makes
    sense in the context of the "helper" function.

    The arguments control the network architecture and training process.
    """

    # create network
    underlying_model = MLP(
        in_size=1024,
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
        pre_flatten=True,
        post_squeeze=True,
        dropout=dropout,
    )
    model = NullTuner(underlying_model).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    # train
    best_epoch, best_val, training_record = train_tunable_model(
        model=model,
        optimizer=opter,
        device=DEVICE,
        n_epochs=n_epochs,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    pred_table = predict(model=model, dataset=eval_dataset, batch_size=1024)

    return best_epoch, best_val, model, training_record, pred_table


def helper(seed: int, **kwargs):
    """Evaluate a model.

    First determines the optimal number of training epochs and then retrains.

    Returns the model, the validation score, and the predictions on the test set.
    """
    torch.manual_seed(seed)
    best_epoch, _, _, _, _ = test_mlp(
            **kwargs
    )

    new_kwargs = kwargs.copy()
    new_kwargs["n_epochs"] = best_epoch

    torch.manual_seed(seed)
    _, _, model, training_report, test_pred = test_mlp(
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
    layer_sel =  [512,1028]
    wdecay = 0.0001
    n_epochs = 300 # this the max number of epochs considered; early stopping is used.
    lr = 1e-4
    dropout = 0.3
    batch_size = 32
    record = {}

    print("Generating dataset...")
    train_dataset, valid_dataset, test_data = get_datasets(device=DEVICE, train_specs=['small_base'], val_specs=['small_base'], feat_type="t5", include_test=True)
    print("Done.")

    for seed in attempt_seeds:
        model, valid, test_pred = helper(
            seed=seed,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            eval_dataset=test_data,
            dropout=dropout,
            batch_size=batch_size,
            hidden_layer_sizes=layer_sel,
            weight_decay=wdecay,
            n_epochs=n_epochs,
            learning_rate=lr,
        )
        record.update({valid: (model, test_pred)})
    best_model, best_table = record[min(record.keys())]
    print("val from random inits:", list(record.keys()))
    core_name = "TESTEVAL_mlp_t5_l{}_wdecay{}_learningrate{}_nepochs{}_dropout{}_batchsize{}_smallbaseonly_nulltuner_multiattempt".format(repr(layer_sel), wdecay, lr, n_epochs, dropout, batch_size)
    best_table.to_csv(core_name+".csv")
    torch.save(best_model, core_name+".pt")


if __name__ == "__main__":
    run()

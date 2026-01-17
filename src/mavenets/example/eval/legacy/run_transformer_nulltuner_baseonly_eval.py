"""Evaluate a base-trained fixed transformer architecture on test sets.

The model is trained and evaluated on the base experiment. No per-experiment
heads are used.
"""
from typing import Final, Tuple, Sequence
import torch
import pandas as pd  # type: ignore
from itertools import product
from random import Random
from ....data import get_datasets, CORE_DATA_SPECS
from ....network import SumTransformer, NullTuner
from ....tools import train_tunable_model
from ....report import predict

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 1000

def test_transformer(
    n_blocks: int,
    n_heads: int,
    embedding_size: int,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
    fan_size: int = 16,
    mha_drop: float = 0.2,
    transformer_mlp_drop: float = 0.2,
    n_final_layers: int = 0,
    final_dropout: float = 0.0,
) -> Tuple:
    """Train model and evaluate on the test set, with quirks.

    A transformer is trained on data from the base experiment. No experimental head is used.

    This function returns a model and a dataframe with the test predictions.
    However, the test predictions are obtained from the model at the end of
    training, which may be overtrained. Early stopping is performed, but the model
    parameters are not rolled back to the optimal epoch after stopping training.

    The arguments control the network architecture and training process.
    """

    # collate base dataset.
    train_dataset, valid_dataset = get_datasets(train_specs=['base'], val_specs=['base'], device=DEVICE)

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE,
        )
        report_datasets.update({spec.name: vdset})

    # create model
    underlying_model = SumTransformer(
        alphabet_size=256,
        n_transformers=n_blocks,
        emb_size=embedding_size,
        n_heads=n_heads,
        block_mlp_dropout=transformer_mlp_drop,
        block_mha_dropout=mha_drop,
        n_final_layers=n_final_layers,
        final_dropout=final_dropout,
    )
    model = NullTuner(underlying_model).to(DEVICE)
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    # train
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
        progress_bar=False,
    )

    # get test data
    _, _, eval_data = get_datasets(device=DEVICE, include_test=True)
    pred_table = predict(model=model, dataset=eval_data, batch_size=1024)

    return model, pred_table


def run() -> None:
    """Evaluate a model with a given set of hypers on test data."""
    n_blocks = 6
    n_heads = 8
    emb_size = 64
    weight_decay = 0.005
    mlp_drop = 0.1
    mha_drop = 0.05
    n_final_layers = 1
    final_dropout = 0.10
    n_epochs = 75 # Model is trained for this many epochs, early stopping is imperfect here.

    model, table = test_transformer(
        n_blocks=n_blocks,
        n_heads=n_heads,
        embedding_size=emb_size,
        weight_decay=weight_decay,
        mha_drop=mha_drop,
        transformer_mlp_drop=mlp_drop,
        n_final_layers=n_final_layers,
        final_dropout=final_dropout,
        n_epochs = n_epochs ,
    )
    core_name = "TESTEVAL_b{}_h{}_e{}_wdecay{}_mlpdrop{}_mhadrop{}_flayers{}_fdrop{}_nepoch{}".format(
        n_blocks,
        n_heads,
        emb_size,
        weight_decay,
        mlp_drop,
        mha_drop,
        n_final_layers,
        final_dropout,
        n_epochs,
    )
    table.to_csv(core_name+".csv")
    torch.save(model, core_name+".pt")


if __name__ == "__main__":
    run()

"""Train transformer using prediction accuracy."""
from typing import Final, Tuple
import torch
import pandas as pd  # type: ignore
from ..data import get_datasets, DATA_SPECS
from ..network import SumTransformer, SharedFanTuner
from ..tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5


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
) -> Tuple[int,float,pd.DataFrame]:
    """Train model and evaluate."""
    # these two seem to be locked together

    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE,
        )
        report_datasets.update({spec.name: vdset})

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

    return results


def weight_scan() -> None:
    """Scan over various weight decay options.

    Prints results and writes csv as it runs.
    """
    WEIGHT_DECAYS: Final = [0.02, 0.05, 0.1]
    N_BLOCKS: Final = 4
    N_HEADS: Final = 8
    EMBEDDING_SIZE: Final = 32
    for w in WEIGHT_DECAYS:
        epoch, val, table = test_transformer(
            n_blocks=N_BLOCKS,
            n_heads=N_HEADS,
            embedding_size=EMBEDDING_SIZE,
            weight_decay=w,
        )
        print(w, epoch, round(val, 3))
        table.to_csv(
            "b{}_h{}_e{}_wdecay{}.csv".format(N_BLOCKS, N_HEADS, EMBEDDING_SIZE, w)
        )


def scan() -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    WEIGHT_DECAY: Final = 0.005
    FINAL_DROPOUT: Final = 0.2
    SCAN_PARAMS: Final = [
        (5, 4, 32, 0.1, 0.05, 1),
        (5, 4, 32, 0.1, 0.05, 2),
        (5, 4, 32, 0.1, 0.05, 0),
        (5, 4, 32, 0.2, 0.1, 1),
        (5, 4, 32, 0.2, 0.1, 2),
        (5, 4, 32, 0.2, 0.1, 0),
        (4, 4, 32, 0.2, 0.1, 1),
        (4, 4, 32, 0.2, 0.1, 2),
        (4, 4, 32, 0.2, 0.1, 0),
        (4, 8, 32, 0.2, 0.1, 1),
        (4, 8, 32, 0.2, 0.1, 2),
        (4, 8, 32, 0.2, 0.1, 0),
    ]
    for (
        n_blocks,
        n_heads,
        emb_size,
        mha_drop,
        mlp_drop,
        n_final_layers,
    ) in SCAN_PARAMS:
        epoch, val, table = test_transformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            embedding_size=emb_size,
            weight_decay=WEIGHT_DECAY,
            mha_drop=mha_drop,
            transformer_mlp_drop=mlp_drop,
            n_final_layers=n_final_layers,
            final_dropout=FINAL_DROPOUT,
        )
        print(
            n_blocks,
            n_heads,
            emb_size,
            WEIGHT_DECAY,
            mlp_drop,
            mha_drop,
            n_final_layers,
            round(val, 3),
        )
        name = "b{}_h{}_e{}_wdecay{}_mlpdrop{}_mhadrop{}_flayers{}_fdrop{}.csv".format(
            n_blocks,
            n_heads,
            emb_size,
            WEIGHT_DECAY,
            mlp_drop,
            mha_drop,
            n_final_layers,
            FINAL_DROPOUT,
        )
        table.to_csv(name)


if __name__ == "__main__":
    scan()

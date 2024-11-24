"""Train message passing graph network using prediction accuracy."""
from typing import Final
import torch
import pandas as pd  # type: ignore
from ..data import get_datasets, DATA_SPECS
from ..network import SharedFanTuner, GraphNet
from ..tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5


def test_mpn(
    n_hidden: int = 128,
    n_messages: int = 6,
    window_size: int = 10,
    n_distance_feats: int = 10,
    n_features: int = 21,
    batch_size: int = 64,
    eval_batch_size: int = int(2**7),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 50,
    grad_clip: int = 300,
    fan_size: int = 16,
    compile: bool = True,
) -> pd.DataFrame:
    """Train model and evaluate."""
    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        feat_type="onehot",
        graph=True,
        graph_sequence_window_size=window_size,
    )

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
            feat_type="onehot",
            graph=True,
            graph_sequence_window_size=window_size,
        )
        report_datasets.update({spec.name: vdset})

    n_edge_feats = (2 * window_size + 1) + n_distance_feats

    underlying_model = GraphNet(201, n_features, n_hidden, n_messages, n_edge_feats).to(
        DEVICE
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
        train_batch_size=batch_size,
        report_datasets=report_datasets,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        compile_mode="max-autotune",
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
        graph=True,
    )

    return results


def scan() -> None:
    """Test various hyperparmeter combinations.

    Currently, only one model is tested due to poor speed.
    """
    for _ in range(1):
        epoch, val, table = test_mpn()


if __name__ == "__main__":
    scan()

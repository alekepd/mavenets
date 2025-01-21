"""Train MLP using prediction accuracy.

Uses embeddings from pretrained t5 as featurization.
"""
from typing import Final, List, Dict
from itertools import product
import torch
from torch.utils.data import Dataset
import pandas as pd  # type: ignore
from ..data import get_datasets, DATA_SPECS
from ..network import MLP, SharedFanTuner
from ..tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5


def test_mlp(
    hidden_layer_sizes: List[int],
    train_dataset: Dataset,
    valid_dataset: Dataset,
    report_datasets: Dict[str, Dataset],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
    fan_size: int = 16,
) -> pd.DataFrame:
    """Train model and evaluate."""
    underlying_model = MLP(
        in_size=1024,  # size of t5 embedding
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
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


def scan() -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    WEIGHT_DECAY: Final = 5e-3
    layer_basis = [16, 32, 64, 128, 256]
    layer_configs = (
        list(product(layer_basis))
        + list(product(layer_basis, layer_basis))
        + list(product(layer_basis, layer_basis, layer_basis))
    )

    print("Generating dataset...")
    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="t5")

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="t5"
        )
        report_datasets.update({spec.name: vdset})

    print("Done.")

    for layer_sel in layer_configs:
        epoch, val, table = test_mlp(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            report_datasets=report_datasets,
            hidden_layer_sizes=list(layer_sel),
            weight_decay=WEIGHT_DECAY,
        )
        print(
            layer_sel,
            round(val, 3),
        )
        name = "mlp_l{}_wdecay{}_t5.csv".format(repr(layer_sel), WEIGHT_DECAY)
        table.to_csv(name)


if __name__ == "__main__":
    scan()

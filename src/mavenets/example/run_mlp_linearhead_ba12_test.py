"""BA12 test of MLP with linear tuner.

Test the performance of a MLP with a linear tuner for extrapolation on
BA12 datasets.
"""
from typing import Final, Sequence
from itertools import product
import torch
import pandas as pd  # type: ignore
from ..data import get_datasets, CORE_DATA_SPECS
from ..network import MLP, LinearTuner
from ..tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5


def test_mlp(
    hidden_layer_sizes: Sequence[int],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 5000,
    grad_clip: int = 300,
) -> pd.DataFrame:
    """Train model and evaluate."""
    non_ba_specs = [x for x in CORE_DATA_SPECS if x.name not in ["BA1","BA2"]]

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=CORE_DATA_SPECS,
        val_specs=CORE_DATA_SPECS,
        feat_type="onehot",
    )

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
        pre_flatten=True,
        post_squeeze=True,
    )
    model = LinearTuner(underlying_model, n_heads=8).to(
        DEVICE
    )
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
        reporting_batch_size=eval_batch_size,
        compile=compile,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )
    print(results[0], results[1])

    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=non_ba_specs,
        val_specs=non_ba_specs,
        feat_type="onehot",
    )

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
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
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )
    print(results[0], results[1])

    return results


def scan() -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    weight_decays: Final = (1e-2,5e-3,1e-3,5e-4,1e-4)
    layer_options = (8, 16, 32, 64, 128, 256)
    for wdecay in weight_decays:
        for layer_sel in (
            list(product(layer_options))
            + list(product(layer_options, layer_options))
            + list(product(layer_options, layer_options, layer_options))
        ):
            _,_, summary = test_mlp(
                hidden_layer_sizes=layer_sel,
                weight_decay=wdecay,
            )
            print(layer_sel,summary.BA1.min(),summary.BA2.min())
            summary.to_csv(str(layer_sel)+"_wdecay{wdecay}_batest.csv")


if __name__ == "__main__":
    scan()

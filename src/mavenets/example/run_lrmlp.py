"""Train transformer using prediction accuracy."""
from typing import Final
from itertools import product
import torch
import pandas as pd  # type: ignore
from ..data import get_datasets, DATA_SPECS
from ..network import LRMLP, SharedFanTuner
from ..tools import train_tunable_model

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 1


def test_lrmlp(
    augment_channel_size: int,
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
    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    underlying_model = LRMLP(
        in_size=21 * 201,
        out_size=1,
        augment_channel_size=augment_channel_size,
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
        progress_bar=True,
    )

    return results


def scan() -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    WEIGHT_DECAYS: Final = [5e-3, 1e-3, 5e-4, 1e-2]
    AUG_SIZES: Final = [1, 2, 3, 4, 8, 16, 32]
    FAN_SIZES: Final = [5, 8, 10, 16, 2, 4]
    for aug_size, fan_size, weight_decay in product(
        AUG_SIZES, FAN_SIZES, WEIGHT_DECAYS
    ):
        epoch, val, table = test_lrmlp(
            augment_channel_size=aug_size,
            fan_size=fan_size,
            weight_decay=weight_decay,
        )
        print(
            aug_size,
            fan_size,
            round(val, 3),
        )
        name = "lrmlp_augsize{}_fansize{}_wdecay{}.csv".format(
            aug_size, fan_size, weight_decay
        )
        table.to_csv(name)


if __name__ == "__main__":
    scan()

"""Train transformer using prediction accuracy."""
from typing import Final, List
import torch
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

    underlying_model = MLP(
        in_size=21 * 201,
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
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
    layer_configs = [
        [16],
        [32],
        [64],
        [128],
        [256],
        [16, 16],
        [32, 16],
        [64, 16],
        [128, 16],
        [256, 16],
        [16, 32],
        [32, 32],
        [64, 32],
        [128, 32],
        [256, 32],
        [16, 64],
        [32, 64],
        [64, 64],
        [128, 64],
        [256, 64],
        [16, 128],
        [32, 128],
        [64, 128],
        [128, 128],
        [256, 128],
        [16, 16, 16],
        [32, 16, 16],
        [64, 16, 16],
        [128, 16, 16],
        [256, 16, 16],
        [16, 32, 16],
        [32, 32, 16],
        [64, 32, 16],
        [128, 32, 16],
        [256, 32, 16],
    ]
    for layer_sel in layer_configs:
        epoch, val, table = test_mlp(
            hidden_layer_sizes=layer_sel,
            weight_decay=WEIGHT_DECAY,
        )
        print(
            layer_sel,
            round(val, 3),
        )
        name = "mlp_l{}_wdecay{}.csv".format(repr(layer_sel), WEIGHT_DECAY)
        table.to_csv(name)


if __name__ == "__main__":
    scan()

"""Train model and create csv of output predictions."""
from typing import Final, Any, Sequence
import torch
from ..data import get_datasets, DATA_SPECS
from ..network import MLP, SharedFanTuner
from ..tools import train_tunable_model
from ..report import predict

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
    n_epochs: int = 30,
    grad_clip: int = 300,
    fan_size: int = 16,
) -> Any:
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

    _ = train_tunable_model(
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

    return model


def scan() -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    WEIGHT_DECAY: Final = 5e-3
    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")
    for layer_sel in [(16, 32, 32)]:
        model = test_mlp(
            hidden_layer_sizes=layer_sel,
            weight_decay=WEIGHT_DECAY,
        )
        val_table = predict(model=model, dataset=valid_dataset)
        val_table.to_csv("validation_predictions.csv")


if __name__ == "__main__":
    scan()

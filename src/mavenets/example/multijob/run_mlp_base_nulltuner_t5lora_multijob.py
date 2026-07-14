"""Train (no fan) mlp with LoRA-finetuned T5 using multiple GPUs.

This model uses LoRA adapters on a pretrained T5 encoder, with a downstream MLP
for regression. The T5 encoder runs in the training forward pass with gradients
flowing through the LoRA parameters.

The model is trained on the 'base' dataset only.

All multijob examples are designed to be run via code similar to that in run_example.py.

The scan function is meant to be called with two arguments. See run_example.py for details.
"""
from typing import Final, List, Sequence, Tuple, Dict
from itertools import product
import torch
from torch.utils.data import Dataset
import pandas as pd  # type: ignore
from random import Random
from pathlib import Path
from ...data import get_datasets, CORE_DATA_SPECS
from ...network import MLP, NullTuner
from ...network.t5lora import T5LoRAModel
from ...tools import train_tunable_model

torch._dynamo.config.cache_size_limit = 2096  # type: ignore

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5
LORA_RANK: Final = 8
LORA_ALPHA: Final = 16


def get_tasks(
    data: Sequence,
    replica: int,
    total_n_replicas: int,
    shuffle: bool = False,
    seed: int = 68712364283,
):
    assert total_n_replicas > 0
    size = len(data)
    if shuffle:
        procced: Sequence = Random(seed).sample(data, size)
    else:
        procced = data
    if total_n_replicas == 1:
        return procced
    else:
        raw_breaks = list(range(0, size, size // total_n_replicas))
        breaks = raw_breaks[:total_n_replicas] + [size]
        assert len(breaks) == total_n_replicas + 1
        return procced[breaks[replica] : breaks[replica + 1]]


def test_mlp(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    report_datasets: Dict[str, Dataset],
    hidden_layer_sizes: List[int],
    compile: bool = False,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    mlp_learning_rate: float = 3e-4,
    lora_learning_rate: float = 1e-4,
    weight_decay: float = 0.005,
    dropout: float = 0.0,
    n_epochs: int = 200,
    grad_clip: int = 300,
    gradient_checkpointing: bool = False,
) -> Tuple[int, float, pd.DataFrame]:
    """Train model and evaluate."""

    mlp = MLP(
        in_size=1024,  # T5 embedding dimension
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
        post_squeeze=True,
        dropout=dropout,
    )

    t5_lora = T5LoRAModel(
        downstream=mlp,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = NullTuner(t5_lora).to(DEVICE)

    # Separate parameter groups: LoRA params get a lower learning rate
    lora_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    opter = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lora_learning_rate},
            {"params": other_params, "lr": mlp_learning_rate},
        ],
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
        progress_bar=False,
    )

    return results


def scan(replica: int, total_n_replicas: int) -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    mlp_lr_ops = (3e-4, 1e-4)
    lora_lr_ops = (1e-4, 5e-5)
    layer_options = (256, 512)
    dropout_opts = (0.0, 0.1)
    layer_ops = (
        list(product(layer_options))
        + list(product(layer_options, layer_options))
    )
    options = list(
        product(layer_ops, mlp_lr_ops, lora_lr_ops, dropout_opts)
    )
    tasks = get_tasks(
        data=options,
        replica=replica,
        total_n_replicas=total_n_replicas,
        shuffle=True,
    )

    print("Generating dataset...")
    train_dataset, valid_dataset = get_datasets(
        device=DEVICE,
        train_specs=["base"],
        val_specs=["base"],
        feat_type="integer",
    )

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
            feat_type="integer",
        )
        report_datasets.update({spec.name: vdset})

    print("Done.")

    for layer_sel, mlp_lr, lora_lr, dropout in tasks:
        name = "mlp_l{}_mlplr{}_loralr{}_dropout{}_base_nulltuner_t5lora_r{}.csv".format(
            repr(layer_sel), mlp_lr, lora_lr, dropout, LORA_RANK
        )
        if Path(name).is_file():
            continue
        epoch, val, table = test_mlp(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            report_datasets=report_datasets,
            hidden_layer_sizes=layer_sel,
            mlp_learning_rate=mlp_lr,
            lora_learning_rate=lora_lr,
            dropout=dropout,
        )
        table.to_csv(name)

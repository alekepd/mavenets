"""Train fan-mlp using prediction accuracy using multiple GPUs.

The model is first trained on the all datasets; then the tuner is extracted, parameters
of the underlying model are re-initialized, and the model is retrained on the non-BA datasets.

All multijob examples are designed to be run via code similar to that in run_example.py.

The scan function is meant to be called with two arguments. See run_example.py for details.
"""
from typing import Final, List, Sequence, Tuple
import torch
import pandas as pd  # type: ignore
from random import Random
from itertools import product
from pathlib import Path
from ...data import get_datasets, CORE_DATA_SPECS
from ...network import MLP, SharedFanTuner
from ...tools import train_tunable_model

torch._dynamo.config.cache_size_limit = 2096  # type: ignore

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 5

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
    hidden_layer_sizes: List[int],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 1000,
    grad_clip: int = 300,
    fan_size: int = 16,
) -> Tuple[int, float, pd.DataFrame]:
    """Train model and evaluate."""

    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")

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
        train_batch_size=batch_size,
        reporting_batch_size=eval_batch_size,
        compile=compile,
        compile_mode="max-autotune",
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=False,
    )

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="onehot"
        )
        report_datasets.update({spec.name: vdset})

    non_ba_specs = [x for x in CORE_DATA_SPECS if x.name not in ["BA1","BA2"]]

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
        compile_mode="max-autotune",
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        start_loss_param=0.0,
        progress_bar=False,
    )

    return results


def scan(replica: int, total_n_replicas: int) -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    learning_rate_ops: Final = [1e-4,3e-4]
    fan_size_ops: Final = [1,2,4,8,16,32,64,128]
    weight_decay_ops: Final = (5e-3,1e-3,5e-4)
    layer_options = (8, 16, 32, 64, 128, 256)
    layer_ops = (list(product(layer_options)) +
                      list(product(layer_options, layer_options)) +
                      list(product(layer_options, layer_options, layer_options)))
    options = list(product(layer_ops, weight_decay_ops, fan_size_ops, learning_rate_ops))
    tasks = get_tasks(data=options,
                      replica=replica,
                      total_n_replicas=total_n_replicas,
                      shuffle=True)

    for layer_sel,wdecay,fsize,lr in tasks:
        name = "mlp_l{}_wdecay{}_fansize{}_lr{}_batest.csv".format(repr(layer_sel),
                                                                    wdecay,
                                                                    fsize,
                                                                    lr)
        if Path(name).is_file():
            continue
        epoch, val, table = test_mlp(
            hidden_layer_sizes=layer_sel,
            weight_decay=wdecay,
            fan_size=fsize,
            learning_rate=lr,
        )
        table.to_csv(name)

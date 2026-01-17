"""Train (no fan) mlp using prediction accuracy using multiple GPUs.

This model uses embeddings from pretrained t5 as featurization.

The model is trained on the 'ba1' dataset only.

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
from ...data import get_datasets, DATA_SPECS
from ...network import MLP, NullTuner
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
    train_dataset: Dataset,
    valid_dataset: Dataset,
    report_datasets: Dict[str, Dataset],
    hidden_layer_sizes: List[int],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    dropout: float = 0.0,
    n_epochs: int = 1000,
    grad_clip: int = 300,
) -> Tuple[int, float, pd.DataFrame]:
    """Train model and evaluate."""

    underlying_model = MLP(
        in_size=1024,  # size of t5 embedding
        out_size=1,
        hidden_sizes=hidden_layer_sizes,
        post_squeeze=True,
        dropout=dropout
    )
    model = NullTuner(underlying_model).to(DEVICE)
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
        progress_bar=False,
    )

    return results


def scan(replica: int, total_n_replicas: int) -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    weight_decay_ops = (5e-3,1e-3,5e-4,1e-4)
    learning_rate_ops = (3e-4,1e-4)
    layer_options = (128, 256, 512, 1028)
    dropout_opts = (0.0, 0.1, 0.3, 0.5)
    layer_ops = (list(product(layer_options)) +
                      list(product(layer_options, layer_options)) +
                      list(product(layer_options, layer_options, layer_options)))
    options = list(product(layer_ops, weight_decay_ops, learning_rate_ops, dropout_opts))
    tasks = get_tasks(data=options,
                      replica=replica,
                      total_n_replicas=total_n_replicas,
                      shuffle=True)

    print("Generating dataset...")
    train_dataset, valid_dataset = get_datasets(device=DEVICE, train_specs=['base'], val_specs=['base'], feat_type="t5")

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec], val_specs=[spec], device=DEVICE, feat_type="t5"
        )
        report_datasets.update({spec.name: vdset})

    print("Done.")

    for layer_sel,wdecay,lr,dropout in tasks:
        name = "mlp_l{}_wdecay{}_lr{}_dropout{}_batch128_base_nulltuner.csv".format(repr(layer_sel), wdecay, lr, dropout)
        if Path(name).is_file():
            continue
        epoch, val, table = test_mlp(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            report_datasets=report_datasets,
            hidden_layer_sizes=layer_sel,
            weight_decay=wdecay,
            learning_rate=lr,
            dropout=dropout,
            batch_size=128,
        )
        table.to_csv(name)

"""Train a fanned transformer using prediction accuracy using multiple GPUs.

The model is trained on all the data.

All multijob examples are designed to be run via code similar to that in run_example.py.

The scan function is meant to be called with two arguments. See run_example.py for details.
"""
from typing import Final, Tuple, Sequence
import torch
import pandas as pd  # type: ignore
from itertools import product
from random import Random
from pathlib import Path
from ...data import get_datasets, CORE_DATA_SPECS
from ...network import SumTransformer, SharedFanTuner
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

    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    report_datasets = {}
    for spec in CORE_DATA_SPECS:
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
        progress_bar=False,
    )

    return results


def scan(replica: int, total_n_replicas: int) -> None:
    """Scan over various hyperparameter choices.

    Prints results and writes csv as it runs.
    """
    assert replica < total_n_replicas
    fan_size_ops: Final = [1,2,4,8,16,32,64,128]
    n_blocks_ops = (1,2,3,4,5,6)
    n_heads_ops = (2,4,8,16)
    emb_size_ops = (16, 32, 64)
    mha_drop_ops = (0.05,0.1,0.2)
    mlp_drop_ops = (0.05,0.1,0.2)
    n_final_layers_ops = (0,1,2,3)
    weight_decay_ops = (5e-3,1e-3,5e-4)
    final_dropout_ops = (0.0,0.05,0.1)
    options = list(product(n_blocks_ops,
                           n_heads_ops,
                           emb_size_ops,
                           mha_drop_ops,
                           mlp_drop_ops,
                           n_final_layers_ops,
                           weight_decay_ops,
                           final_dropout_ops,
                           fan_size_ops))
    tasks = get_tasks(data=options,
                      replica=replica,
                      total_n_replicas=total_n_replicas,
                      shuffle=True)
    for (n_blocks,
         n_heads,
         emb_size,
         mha_drop,
         mlp_drop,
         n_final_layers,
         weight_decay,
         final_dropout,
         fan_size,
    ) in tasks:
        name = "b{}_h{}_e{}_wdecay{}_mlpdrop{}_mhadrop{}_flayers{}_fdrop{}_fansize{}.csv".format(
            n_blocks,
            n_heads,
            emb_size,
            weight_decay,
            mlp_drop,
            mha_drop,
            n_final_layers,
            final_dropout,
            fan_size,
        )
        if Path(name).is_file():
            continue
        epoch, val, table = test_transformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            embedding_size=emb_size,
            weight_decay=weight_decay,
            mha_drop=mha_drop,
            transformer_mlp_drop=mlp_drop,
            n_final_layers=n_final_layers,
            final_dropout=final_dropout,
            fan_size=fan_size,
        )
        table.to_csv(name)

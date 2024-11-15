"""Train transformer using prediction accuracy."""
from typing import Final, Tuple, Any
import torch
from torch import nn
from ..data import get_datasets, DATA_SPECS, get_default_int_encoder, SARS_COV2_SEQ
from ..network import SumTransformer, SharedFanTuner
from ..tools import train_tunable_model
from ..sample import BiasedIntMutate, MetSim

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 10


def get_transformer(
    n_blocks: int = 5,
    n_heads: int = 4,
    embedding_size: int = 32,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 175,
    grad_clip: int = 300,
    fan_size: int = 16,
    mha_drop: float = 0.1,
    transformer_mlp_drop: float = 0.05,
    n_final_layers: int = 1,
    final_dropout: float = 0.01,
) -> Tuple[nn.Module, int, float]:
    """Train model and evaluate."""
    # these two seem to be locked together

    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
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

    report = train_tunable_model(
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
        train_bfloat16=True,
        grad_clip=grad_clip,
        report_stride=REPORT_STRIDE,
        progress_bar=True,
    )

    return underlying_model, report[0], report[1]


def test_sim(beta: float = -10000.0) -> Any:
    """Trains a transformer and runs a simulation with it."""
    BIAS: Final = 0.7
    N_SIM_STEPS: Final = 200000

    # get trained model
    raw_model, best_epoch, best_val = get_transformer()
    print("best epoch", best_epoch)
    print("best score", best_val)

    # run simulation
    with torch.no_grad():
        # create starting sequence
        enc = get_default_int_encoder()
        center_seq = enc.encode(SARS_COV2_SEQ, tensor=True).to(DEVICE)
        # create move generation mechanism
        mut = BiasedIntMutate(0, 20, bias=BIAS, center=center_seq)
        # create simulation
        sim = MetSim(model=raw_model, proposer=mut, beta=beta, compile=True)
        # run simulation
        frames = sim.run(N_SIM_STEPS, device=DEVICE)
    return frames


if __name__ == "__main__":
    test_sim()

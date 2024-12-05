"""Train transformer using prediction accuracy."""
from typing import Final, Tuple, Any, Sequence
import torch
from torch import nn
import pandas as pd
from ..data import (
    get_datasets,
    DATA_SPECS,
    get_default_int_encoder,
    SARS_COV2_SEQ,
    int_to_floatonehot,
)
from ..network import MLP, SharedFanTuner
from ..tools import train_tunable_model
from ..sample import BiasedIntMutate, MetSim

torch.manual_seed(1337)
# tensor cores on
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
REPORT_STRIDE: Final = 10


def get_mlp(
    hidden_layer_sizes: Sequence[int],
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 175,
    grad_clip: int = 300,
    fan_size: int = 16,
) -> Tuple[nn.Module, int, float]:
    """Train model and evaluate."""
    # these two seem to be locked together

    train_dataset, valid_dataset = get_datasets(device=DEVICE,feat_type='onehot')

    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
            feat_type='onehot',
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


def test_sim(beta: float = -1000.0) -> Any:
    """Trains a transformer and runs a simulation with it."""
    # bias is between 0 and 1. 1 forces the simulation to stay at the native state,
    # 0 does not put any bias in.
    BIAS: Final = 0.85

    # length of simulation
    N_SIM_STEPS: Final = 100000

    # BETA IS AN ARGUMENT TO THIS FUNCTION! If you want to look for sequences with a
    # _high_ energy, it should be a negative number. Physical intuition then comes by
    # thinking about it with its sign flipped as a "temperature", so very negative
    # values mean a smoother landscape (analogous to a higher temperature).

    # get trained model
    raw_model, best_epoch, best_val = get_mlp(hidden_layer_sizes=(16, 64), n_epochs=30)
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

        # the MLP is defined to operate on one-hot encodings, but the simulation
        # operates on integer encodings. We wrap the MLP with a featurizer to
        # make it work on integer encodings.
        def _wrapped_model(x: torch.Tensor) -> torch.Tensor:
            return raw_model(int_to_floatonehot(x,21))

        sim = MetSim(model=_wrapped_model, proposer=mut, beta=beta, compile=True)
        # run simulation
        frames = sim.run(N_SIM_STEPS, device=DEVICE)

    # decode the observed sequences from integers to strings
    sequences = enc.batch_decode([x.sequence for x in frames])

    # create table summarizing results
    df = pd.DataFrame([list(x) for x in sequences])
    df.columns = ['p'+str(x) for x in df.columns]
    df['time'] = [x.index for x in frames]
    df['energy'] = [x.energy for x in frames]

    return df


if __name__ == "__main__":
    test_sim()

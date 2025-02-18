"""Train a transformer and run MCMC simulations.

This example trains a transformer from scratch and then uses it for simulation.
It could be adapted to instead load a pre-trained model and use that for
simulation. The simulation is done on the raw (non-tuned) output of the transformer,
but training is done using tuning heads.
"""
from typing import Final, Tuple
import torch
from torch import nn
import pandas as pd #type: ignore
from ..data import get_datasets, DATA_SPECS, get_default_int_encoder, SARS_COV2_SEQ
from ..network import SumTransformer, SharedFanTuner
from ..tools import train_tunable_model
from ..sample import BiasedIntMutate, MetSim

torch.manual_seed(1337)
# This accelerates computations on certain classes of GPUs at the cost of
# some accuracy. Testing suggests that the loss of accuracy is negligible.
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
# controls how often we check the performance of the model on all the present
# data. Lower values provide more detailed validation curves but slow down
# training.
REPORT_STRIDE: Final = 10


def get_transformer(
    n_blocks: int = 5,
    n_heads: int = 4,
    embedding_size: int = 32,
    compile: bool = True,
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 6e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 175,
    grad_clip: int = 300,
    fan_size: int = 8,
    mha_drop: float = 0.1,
    transformer_mlp_drop: float = 0.05,
    n_final_layers: int = 2,
    final_dropout: float = 0.1,
) -> Tuple[nn.Module, int, float]:
    """Train a transformer and return it.

    Note that per-experiment heads are using during training, but the underlying network
    without any tuning heads is returned. This is what the simulation is then run on.
    Alternatively, one could wrap the model with the tuning heads intact and fix the
    tuning head used.

    Arguments specify training options for the transformer.

    Returns:
    -------
    A tuple; first element is the trained model, second is the optimal epoch
    found during training, third is the score at the optimal epoch. Note that the
    model is not necessarily the one from the optimal epoch, but the model
    obtained after the halting training.

    """
    # get data that is used for training and validation
    train_dataset, valid_dataset = get_datasets(device=DEVICE)

    # get data that is used for additional reporting in validation curve.
    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
        )
        report_datasets.update({spec.name: vdset})

    # create transformer
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
    # create the tuner on top of the MLP
    model = SharedFanTuner(underlying_model, n_heads=8, fan_size=fan_size).to(DEVICE)

    # create optimizer
    opter = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        fused=True,
        weight_decay=weight_decay,
    )

    # train the model with tuning heads
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

    # now return the model _WITHOUT_ the tuning part. We also return basic information
    # on the training run.

    return underlying_model, report[0], report[1]


def test_sim(beta: float = -10.0) -> pd.DataFrame:
    """Trains an MLP and runs a simulation with it.

    Arguments:
    ---------
    beta:
        distribution targeted is proportional to exp(-beta U(x)) (plus the native bias) 
        where U is the neural network. If we want _high_ U, beta should be negative.

    Returns:
    -------
    A DataFrame containing the results of the simulation: amino acid choices for
    each position, the energy of each recorded sequence, and the simultation time step 
    at which the sequence was found.

    """
    # bias is between 0 and 1. 1 forces the simulation to stay at the native state,
    # 0 does not put any bias in.
    NATIVE_BIAS: Final = 0.85

    # length of simulation
    N_SIM_STEPS: Final = 100000

    # beta IS AN ARGUMENT TO THIS FUNCTION! If you want to look for sequences with a
    # _high_ energy, it should be a negative number. Physical intuition then comes by
    # thinking about it with its sign flipped as a "temperature", so very negative
    # values mean a smoother landscape (analogous to a higher temperature).

    # train single model for a selected architecture.
    # raw_model is the model without the tuning head.
    raw_model, best_epoch, best_val = get_transformer(n_epochs=125)
    print("best epoch", best_epoch)
    print("best score", best_val)

    with torch.no_grad():
        # create starting sequence, encode to integer form
        enc = get_default_int_encoder()
        center_seq = enc.encode(SARS_COV2_SEQ, tensor=True).to(DEVICE)

        # create move generation mechanism. This is where the native
        # bias is introduced.
        # note that we use 20, not 21, as the 21st symbol is X, representing
        # an unknown amino acid. We probably don't want to sample that.
        mut = BiasedIntMutate(0, 20, bias=NATIVE_BIAS, center=center_seq)

        # create simulation
        sim = MetSim(model=raw_model, proposer=mut, beta=beta, compile=True)

        # run simulation; frames contains the result
        frames = sim.run(N_SIM_STEPS, device=DEVICE)

    # frames is a list of 

    # decode the observed sequences from integers to strings
    sequences = enc.batch_decode([x.sequence for x in frames])

    # create table summarizing results
    df = pd.DataFrame([list(x) for x in sequences])
    df.columns = ["p" + str(x) for x in df.columns]
    df["time"] = [x.index for x in frames]
    df["energy"] = [x.energy for x in frames]

    return df


if __name__ == "__main__":
    test_sim()

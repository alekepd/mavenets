"""Train and MLP and run MCMC simulations.

This example trains a MLP from scratch and then uses it for simulation. It could be 
adapted to instead load a pre-trained model and use that for simulation. The simulation
is performed setting the experiment head to 0, but comments show how to use
the model prior to head calibration as well.
"""
from typing import Final, Tuple, Sequence
import torch
from torch import nn
import pandas as pd  # type: ignore
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
# This accelerates computations on certain classes of GPUs at the cost of
# some accuracy. Testing suggests that the loss of accuracy is negligible.
torch.set_float32_matmul_precision("high")

DEVICE: Final = "cuda"
# controls how often we check the performance of the model on all the present
# data. Lower values provide more detailed validation curves but slow down
# training.
REPORT_STRIDE: Final = 10


def get_mlp(
    hidden_layer_sizes: Sequence[int],  # MLP architecture
    compile: bool = True,  # whether to use torch compile
    batch_size: int = 32,
    eval_batch_size: int = int(2**11),
    learning_rate: float = 3e-4,
    weight_decay: float = 0.005,
    n_epochs: int = 175,  # number of epochs to train for
    grad_clip: int = 300,
    fan_size: int = 16,  # hyperparameter of tuning heads
) -> Tuple[nn.Module, int, float]:
    """Train and MLP and return it.

    Note that per-experiment heads are using during training.

    Arguments specify training options for the MLP.

    Returns:
    -------
    A tuple; first element is the trained model, second is the optimal epoch
    found during training, third is the score at the optimal epoch. Note that the
    model is not necessarily the one from the optimal epoch, but the model
    obtained after the halting training.

    """
    # get data that is used for training and validation
    train_dataset, valid_dataset = get_datasets(device=DEVICE, feat_type="onehot")

    # get data that is used for additional reporting in validation curve.
    report_datasets = {}
    for spec in DATA_SPECS:
        _, vdset = get_datasets(
            train_specs=[spec],
            val_specs=[spec],
            device=DEVICE,
            feat_type="onehot",
        )
        report_datasets.update({spec.name: vdset})

    # create MLP
    underlying_model = MLP(
        in_size=21 * 201,  # this is the size of our alphabet times the sequence size.
        out_size=1,  #
        hidden_sizes=hidden_layer_sizes,  # controls the arch
        pre_flatten=True,
        post_squeeze=True,
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

    # now return the multi-head model and basic reports.

    return model, report[0], report[1]


def test_sim(beta: float = -1000.0, use_exp_zero: bool = True) -> pd.DataFrame:
    """Trains an MLP and runs a simulation with it.

    Arguments:
    ---------
    beta:
        distribution targeted is proportional to exp(-beta U(x)) (plus the native bias)
        where U is the neural network. If we want _high_ U, beta should be negative.
    use_exp_zero:
        Multihead models require a sequence and experiment id to provide an answer.
        However, MCMC only operates on sequence. If this option is True, we force
        the experiment index to be 0 during the MCMC. If False, we instead perform
        MCMC using the shared-across-experiment prediction (before calibration).

    Returns:
    -------
    A DataFrame containing the results of the simulation: amino acid choices for
    each position, the energy of each recorded sequence, and the simultation time step
    at which the sequence was found.

    """
    # bias is between 0 and 1. 1 forces the simulation to stay at the native state,
    # 0 does not put any bias in.
    NATIVE_BIAS: Final = 0.95

    # length of simulation
    N_SIM_STEPS: Final = 5000000

    # maximum number of mutations to allow in the simulation.
    MAX_NUM_MUTATIONS: Final = 9

    # beta IS AN ARGUMENT TO THIS FUNCTION! If you want to look for sequences with a
    # _high_ energy, it should be a negative number. Physical intuition then comes by
    # thinking about it with its sign flipped as a "temperature", so very negative
    # values mean a smoother landscape (analogous to a higher temperature).

    # train single model for a selected architecture.
    model, best_epoch, best_val = get_mlp(hidden_layer_sizes=(32,), n_epochs=30)
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

        if use_exp_zero:
            # this creates a wrapped model that forces the selected calibration
            # head (usually determined by experiment) to be 0
            fixed_model = model.create_singlehead_model(head_index=0)

            # the MLP we trained is defined to operate on one-hot encodings, but
            # the simulation operates on integer encodings. We wrap the MLP with a
            # featurizer to make it work on integer encodings.
            def _wrapped_model(x: torch.Tensor) -> torch.Tensor:
                # int_to_floatonehot converts from integers to onehots
                # even though we don't want to sample the 21st symbol
                # the MLP has the dimensionality for it, so we use 21 here.
                return fixed_model(int_to_floatonehot(x, 21))

        else:
            # instead of wrapping by fixing the head index, we extract the portion
            # of the model before the calibration heads
            raw_model = model.base_model

            # the MLP we trained is defined to operate on one-hot encodings, but
            # the simulation operates on integer encodings. We wrap the MLP with a
            # featurizer to make it work on integer encodings.
            def _wrapped_model(x: torch.Tensor) -> torch.Tensor:
                # int_to_floatonehot converts from integers to onehots
                # even though we don't want to sample the 21st symbol
                # the MLP has the dimensionality for it, so we use 21 here.
                return raw_model(int_to_floatonehot(x, 21))

        # create simulation. This is where the maximum mutation count
        # is given. Note that here the mutations are calculated relative to the
        # wild type (sample as the bias) but these could be calculated with
        # respect to different sequences.
        sim = MetSim(
            model=_wrapped_model,
            proposer=mut,
            beta=beta,
            compile=True,
            center=center_seq,
            max_distance_to_center=MAX_NUM_MUTATIONS,
        )

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

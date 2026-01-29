"""Timing comparison of one-hot vs T5 featurization and MLP inference.

This script benchmarks two featurization approaches used in the mavenets
codebase: one-hot encoding and T5 (ProtTrans) embedding. Synthetic protein
sequences resembling the SARS-CoV2 RBD are generated and featurized using
routines from mavenets.data.featurize.

For each featurization, a randomly initialised 3-layer MLP (matching codebase
conventions) is evaluated so that the cost of a full featurize-then-infer pass
can be understood.

Usage:
    python -m mavenets.examples.time_featurization [--device DEVICE]
                                                    [--n-repeats N]
                                                    [--batch-sizes B1,B2,...]
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List

import torch
from torch import Tensor

from mavenets.data.featurize.core import (
    BASE_ALPHA,
    IntEncoder,
    get_default_int_encoder,
    int_to_floatonehot,
)
from mavenets.data.featurize.t5 import T5EncoderWrapper
from mavenets.data.spec import SARS_COV2_SEQ
from mavenets.network import MLP

# Length of the wild-type sequence used throughout the codebase.
SEQ_LEN = len(SARS_COV2_SEQ)
NUM_CLASSES = len(BASE_ALPHA)  # 21 amino acid symbols

# MLP hidden layer sizes (3 hidden layers, matching typical codebase configs).
HIDDEN_SIZES: List[int] = [128, 64, 32]


# -- helpers ------------------------------------------------------------------


def generate_sequences(n: int, seq_len: int, alphabet: List[str]) -> List[str]:
    """Generate random protein sequences of a fixed length.

    Arguments:
    ---------
    n:
        Number of sequences to generate.
    seq_len:
        Length of each sequence.
    alphabet:
        Characters to sample from.

    Returns:
    -------
    List of n strings, each of length seq_len.

    """
    return ["".join(random.choices(alphabet, k=seq_len)) for _ in range(n)]


def integer_encode_batch(
    sequences: List[str],
    encoder: IntEncoder,
    device: str,
) -> Tensor:
    """Integer-encode a batch of sequences and move to device.

    Mirrors the first step in _process_table from mavenets.data.load.

    Arguments:
    ---------
    sequences:
        Protein sequences to encode.
    encoder:
        IntEncoder instance.
    device:
        Torch device string.

    Returns:
    -------
    Integer-encoded tensor of shape (len(sequences), seq_len).

    """
    return encoder.batch_encode(sequences).to(device)


def featurize_onehot(int_encoded: Tensor) -> Tensor:
    """One-hot encode an integer-encoded batch.

    Mirrors the one-hot path in _process_table.

    Arguments:
    ---------
    int_encoded:
        Integer-encoded tensor (batch, seq_len).

    Returns:
    -------
    Float one-hot tensor (batch, seq_len, NUM_CLASSES).

    """
    return int_to_floatonehot(int_encoded, num_classes=NUM_CLASSES)


def featurize_t5(int_encoded: Tensor, wrapper: T5EncoderWrapper) -> Tensor:
    """Embed sequences using the T5 encoder.

    Mirrors the T5 path in _process_table.

    Arguments:
    ---------
    int_encoded:
        Integer-encoded tensor (batch, seq_len).
    wrapper:
        Initialised T5EncoderWrapper.

    Returns:
    -------
    Embedding tensor (batch, 1024) when per_protein=True.

    """
    return wrapper.batch_encode(int_encoded)


def build_onehot_mlp(device: str) -> MLP:
    """Create a randomly initialised MLP for one-hot features.

    Matches the pattern in mavenets.example.run_mlp: input is a flattened
    one-hot tensor of shape (batch, seq_len * NUM_CLASSES).

    Arguments:
    ---------
    device:
        Torch device string.

    Returns:
    -------
    MLP on the given device in eval mode.

    """
    model = MLP(
        in_size=NUM_CLASSES * SEQ_LEN,
        out_size=1,
        hidden_sizes=HIDDEN_SIZES,
        pre_flatten=True,
        post_squeeze=True,
    )
    model.to(device)
    model.eval()
    return model


def build_t5_mlp(device: str) -> MLP:
    """Create a randomly initialised MLP for T5 features.

    Matches the pattern in mavenets.example.run_t5_mlp: input is a 1024-dim
    per-protein embedding.

    Arguments:
    ---------
    device:
        Torch device string.

    Returns:
    -------
    MLP on the given device in eval mode.

    """
    model = MLP(
        in_size=1024,
        out_size=1,
        hidden_sizes=HIDDEN_SIZES,
        post_squeeze=True,
    )
    model.to(device)
    model.eval()
    return model


# -- benchmarking -------------------------------------------------------------


def time_fn(
    label: str,
    fn: object,
    n_repeats: int,
    device: str,
) -> float:
    """Time a callable, synchronising CUDA if needed.

    Returns the mean elapsed wall-clock time in seconds.

    Arguments:
    ---------
    label:
        Human-readable description printed to stdout.
    fn:
        Zero-argument callable to benchmark.
    n_repeats:
        Number of times to call fn.
    device:
        Torch device string; used to decide whether to call cuda.synchronize.

    Returns:
    -------
    Mean wall-clock time in seconds.

    """
    use_cuda = device.startswith("cuda")

    # warmup
    if callable(fn):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(n_repeats):
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        if callable(fn):
            fn()
        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    print(
        f"  {label}: {mean_time:.6f}s +/- {std_time:.6f}s  "
        f"(n={n_repeats}, total={sum(times):.4f}s)"
    )
    return mean_time


def run_benchmark(
    batch_sizes: List[int],
    n_repeats: int,
    device: str,
) -> None:
    """Run the full benchmark for one-hot and T5 featurization.

    Arguments:
    ---------
    batch_sizes:
        List of batch sizes to benchmark.
    n_repeats:
        Number of timing repetitions per measurement.
    device:
        Torch device string.

    """
    encoder = get_default_int_encoder()

    print("Loading T5 model (one-time cost)...")
    t0 = time.perf_counter()
    t5_wrapper = T5EncoderWrapper(
        integer_encoder=encoder,
        device=device,
        per_protein=True,
    )
    t5_load_time = time.perf_counter() - t0
    print(f"T5 model loaded in {t5_load_time:.2f}s\n")

    print("Building MLPs...")
    onehot_mlp = build_onehot_mlp(device)
    t5_mlp = build_t5_mlp(device)
    print(
        f"  One-hot MLP: in={NUM_CLASSES * SEQ_LEN}, "
        f"hidden={HIDDEN_SIZES}, out=1"
    )
    print(f"  T5 MLP:      in=1024, hidden={HIDDEN_SIZES}, out=1\n")

    print(f"Sequence length : {SEQ_LEN}")
    print(f"Alphabet size   : {NUM_CLASSES}")
    print(f"Device          : {device}")
    print(f"Repeats         : {n_repeats}")
    print()

    for batch_size in batch_sizes:
        print(f"--- batch_size = {batch_size} ---")

        sequences = generate_sequences(batch_size, SEQ_LEN, BASE_ALPHA)

        # Shared: integer encoding
        time_fn(
            "Integer encoding (CPU)",
            lambda: integer_encode_batch(sequences, encoder, "cpu"),
            n_repeats,
            "cpu",
        )

        int_encoded_cpu = integer_encode_batch(sequences, encoder, "cpu")
        int_encoded_dev = int_encoded_cpu.to(device)

        # ── one-hot featurization ────────────────────────────────────────
        # One-hot is extremely fast.  To get a measurable signal we time
        # many iterations inside a single timing call and report per-call
        # time.
        onehot_inner_repeats = 1000
        onehot_mean = time_fn(
            f"One-hot encoding (x{onehot_inner_repeats} inner iters, on {device})",
            lambda: [
                featurize_onehot(int_encoded_dev)
                for _ in range(onehot_inner_repeats)
            ],
            n_repeats,
            device,
        )
        per_call_onehot = onehot_mean / onehot_inner_repeats
        print(f"    -> per-call one-hot: {per_call_onehot:.9f}s")

        # ── T5 featurization ────────────────────────────────────────────
        t5_mean = time_fn(
            f"T5 encoding (on {device})",
            lambda: featurize_t5(int_encoded_cpu, t5_wrapper),
            n_repeats,
            device,
        )

        if per_call_onehot > 0:
            ratio = t5_mean / per_call_onehot
            print(f"    -> T5 / one-hot ratio: {ratio:.1f}x slower")

        # ── MLP inference (forward only) ─────────────────────────────────
        onehot_features = featurize_onehot(int_encoded_dev)
        t5_features = featurize_t5(int_encoded_cpu, t5_wrapper).to(device)

        with torch.no_grad():
            mlp_onehot_inner = 1000
            mlp_onehot_mean = time_fn(
                f"MLP forward on one-hot (x{mlp_onehot_inner} inner iters)",
                lambda: [
                    onehot_mlp(onehot_features)
                    for _ in range(mlp_onehot_inner)
                ],
                n_repeats,
                device,
            )
            per_call_mlp_onehot = mlp_onehot_mean / mlp_onehot_inner
            print(f"    -> per-call MLP(one-hot): {per_call_mlp_onehot:.9f}s")

            mlp_t5_inner = 1000
            mlp_t5_mean = time_fn(
                f"MLP forward on T5 (x{mlp_t5_inner} inner iters)",
                lambda: [
                    t5_mlp(t5_features) for _ in range(mlp_t5_inner)
                ],
                n_repeats,
                device,
            )
            per_call_mlp_t5 = mlp_t5_mean / mlp_t5_inner
            print(f"    -> per-call MLP(T5): {per_call_mlp_t5:.9f}s")

        # ── end-to-end: featurize + MLP forward ─────────────────────────
        print()
        print(f"  End-to-end (featurize + MLP forward), batch_size={batch_size}:")

        with torch.no_grad():
            e2e_onehot_inner = 1000
            e2e_onehot_mean = time_fn(
                f"One-hot + MLP (x{e2e_onehot_inner} inner iters)",
                lambda: [
                    onehot_mlp(featurize_onehot(int_encoded_dev))
                    for _ in range(e2e_onehot_inner)
                ],
                n_repeats,
                device,
            )
            per_call_e2e_onehot = e2e_onehot_mean / e2e_onehot_inner
            print(f"    -> per-call one-hot+MLP: {per_call_e2e_onehot:.9f}s")

            e2e_t5_mean = time_fn(
                f"T5 + MLP (on {device})",
                lambda: t5_mlp(
                    featurize_t5(int_encoded_cpu, t5_wrapper).to(device)
                ),
                n_repeats,
                device,
            )
            print(f"    -> per-call T5+MLP: {e2e_t5_mean:.9f}s")

        if per_call_e2e_onehot > 0:
            e2e_ratio = e2e_t5_mean / per_call_e2e_onehot
            print(f"    -> end-to-end T5 / one-hot ratio: {e2e_ratio:.1f}x slower")

        print()


# -- CLI entry point ----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Time one-hot vs T5 protein sequence featurization and MLP inference."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of timing repetitions per measurement (default: 5).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8,32,128",
        help="Comma-separated batch sizes to benchmark (default: 1,8,32,128).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    run_benchmark(
        batch_sizes=batch_sizes,
        n_repeats=args.n_repeats,
        device=args.device,
    )


if __name__ == "__main__":
    main()

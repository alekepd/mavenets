"""Report explained variance ratio per PCA component for T5 per-residue embeddings.

Loads the 'base' training data, encodes it with T5 (without pooling), and
incrementally fits PCA on the per-residue embeddings. Prints and saves the
explained variance ratio for each component.

Both per-residue and global PCA modes are supported via the --mode flag.

Usage:
    python -m mavenets.example.t5_pca_variance [--n_components N] [--mode MODE] [--device DEVICE] [--output FILE]
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from ..data.featurize.core import get_default_int_encoder
from ..data.featurize.t5 import T5EncoderWrapper
from ..data.featurize.transform import IncrementalPCATransform
from ..data.load import _get_aggregate_mave_csv, SEQ_CNAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report PCA explained variance on T5 per-residue embeddings."
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Number of PCA components to fit (default: 50).",
    )
    parser.add_argument(
        "--mode",
        choices=["per_residue", "global"],
        default="per_residue",
        help="PCA mode: per_residue or global (default: per_residue).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for T5 inference (default: cuda).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for T5 encoding (default: 32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path. If not specified, prints to stdout only.",
    )
    parser.add_argument(
        "--parent_path",
        type=str,
        default="",
        help="Path to data directory containing CSV files (default: current dir).",
    )
    args = parser.parse_args()

    per_residue = args.mode == "per_residue"
    parent_path = Path(args.parent_path)
    enc = get_default_int_encoder()

    # Load 'base' training data
    print("Loading base training data...", file=sys.stderr)
    train_frame = _get_aggregate_mave_csv(
        specs=["base"], identifier="train_filename", directory=parent_path
    )
    int_encoded = enc.batch_encode(train_frame.loc[:, SEQ_CNAME])
    n = int_encoded.shape[0]
    print(f"Loaded {n} training sequences.", file=sys.stderr)

    # Create T5 encoder (no pooling, no flattening)
    print("Loading T5 model...", file=sys.stderr)
    t5 = T5EncoderWrapper(
        integer_encoder=enc,
        device=args.device,
        per_protein=False,
        flatten=False,
        batch_size=args.batch_size,
    )

    # Create PCA transform
    pca = IncrementalPCATransform(
        n_components=args.n_components,
        per_residue=per_residue,
    )

    # Incrementally fit PCA on T5 embeddings
    print("Fitting PCA on T5 embeddings...", file=sys.stderr)
    batch_size = args.batch_size
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        piece = int_encoded[start:end]
        embeddings = t5.vectorized_encode(piece).cpu()
        pca.partial_fit_chunk(embeddings)
        print(
            f"  Processed {end}/{n} sequences",
            file=sys.stderr,
            end="\r",
        )
    print(file=sys.stderr)

    # Extract results
    import numpy as np

    variance_ratio: np.ndarray[object, np.dtype[np.float64]] = pca._pca.explained_variance_ratio_  # type: ignore[reportAssignmentType]
    cumulative: np.ndarray[object, np.dtype[np.float64]] = variance_ratio.cumsum()

    # Print results
    print(f"\nPCA mode: {args.mode}")
    print(f"Components: {args.n_components}")
    print(f"Total variance captured: {cumulative[-1]:.6f}")
    print()
    print(f"{'component':>10}  {'variance_ratio':>16}  {'cumulative':>12}")
    print("-" * 44)
    for i, (vr, cum) in enumerate(zip(variance_ratio, cumulative), start=1):
        print(f"{i:>10}  {vr:>16.8f}  {cum:>12.8f}")

    # Optionally save to CSV
    if args.output is not None:
        output_path = Path(args.output)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["component", "variance_ratio", "cumulative"])
            for i, (vr, cum) in enumerate(
                zip(variance_ratio, cumulative), start=1
            ):
                writer.writerow([i, f"{vr:.10f}", f"{cum:.10f}"])
        print(f"\nResults saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    with torch.no_grad():
        main()

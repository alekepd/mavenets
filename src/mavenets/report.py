"""Routines for creating predictions using trained models."""
from pathlib import Path
from typing import TypeVar, Final, List, Union
import torch
import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as pygDataLoader  # type: ignore
import pandas as pd  # type: ignore
from .network import MHTuner
from .tools import SIGNAL_PYGBATCHKEY, EXP_PYGBATCHKEY
from .data import resolve_dataspec, SequenceDataset, SARS_COV2_SEQ

_T = TypeVar("_T")

REFERENCE_KEY: Final = "reference"
TUNED_PRED_KEY: Final = "tuned"
RAW_PRED_KEY: Final = "raw"
EXPID_KEY: Final = "experiment"
SEQUENCE_KEY: Final = "sequence"
MUTCOUNT_KEY: Final = "mutations_from_sarscov2"
TUNED_CALIBRATED_KEY: Final = "tuned_calibrated"
RAW_CALIBRATED_KEY: Final = "raw_calibrated"


def _compute_mutation_distances(sequences: List[str], reference: str) -> List[int]:
    """Compute the number of mutations from a reference sequence for each sequence.

    Arguments:
    ---------
    sequences:
        List of amino acid sequences to compare.
    reference:
        Reference sequence to compare against.

    Returns:
    -------
    List of integers, each representing the number of positions where the
    corresponding sequence differs from the reference.

    """
    distances = []
    for seq in sequences:
        if len(seq) != len(reference):
            raise ValueError(
                f"Sequence length ({len(seq)}) does not match "
                f"reference length ({len(reference)})"
            )
        distance = sum(1 for a, b in zip(seq, reference) if a != b)
        distances.append(distance)
    return distances


def predict(
    model: MHTuner,
    dataset: Dataset,  # type: ignore[type-arg]
    graph: bool = False,
    translate_experiment_ids: bool = True,
    batch_size: int = 256,
    linear_calibration: bool = False,
) -> pd.DataFrame:
    """Create a table of raw and tuned predictions.

    model and dataset must already reside on the same computational device.

    Arguments:
    ---------
    model:
        Trained MDTuner model. Note that this routing is not compartible with all
        torch.Modules, as we extract both tuned and non-tuned output.
    dataset:
        torch.Dataset containing data for evaluation. Should be of the same format
        as those used for training. If a SequenceDataset is provided, the returned
        DataFrame will include additional columns for sequences and mutation counts.
    graph:
        If model operates on pyg-style batches, this must be set to True.
    translate_experiment_ids:
        If True, integer labels of heads are translated to the string names of
        experiments.
    batch_size:
        Batch size to use when evaluating the predictions.
    linear_calibration:
        If True, two additional columns are added to the returned DataFrame:
        "tuned_calibrated" and "raw_calibrated". These are created by fitting
        a linear least squares model (per experiment) from the prediction column
        to the reference column, then applying that model to produce calibrated
        values.

    Returns:
    -------
    pd.DataFrame with the following columns:
    "reference"
        Reference value that was likely used as a training target.
    "tuned"
        Tuned prediction for a given sequence (i.e., passed through the tuning layer).
    "raw"
        Untuned prediction for a given sequence.
    "experiment"
        integer denoting which tuner head was used for this prediction.

    If linear_calibration is True, the following additional columns are included:
    "tuned_calibrated"
        Tuned predictions linearly calibrated to the reference, per experiment.
    "raw_calibrated"
        Raw predictions linearly calibrated to the reference, per experiment.

    If dataset is a SequenceDataset, the following additional columns are included:
    "sequence"
        The raw amino acid sequence.
    "mutations_from_sarscov2"
        Number of mutations from the SARS-CoV-2 reference sequence.

    """
    if graph:
        loader = pygDataLoader(
            dataset,
            batch_size=batch_size,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
        )

    model.eval()

    raw_predictions = []
    tuned_predictions = []
    experiments = []
    references = []

    with torch.no_grad():
        # grab batch, evaluate, record
        for batch in loader:
            if graph:
                full_inp = batch
                signal = batch[SIGNAL_PYGBATCHKEY]
                dataset_index = batch[EXP_PYGBATCHKEY]
                tuned_pred, raw_pred = model(
                    inp=full_inp, head_index=dataset_index, return_raw=True
                )
            else:
                inp, signal, dataset_index = batch
                tuned_pred, raw_pred = model(
                    inp=inp, head_index=dataset_index, return_raw=True
                )
            references.append(signal.numpy(force=True))
            experiments.append(dataset_index.numpy(force=True))
            tuned_predictions.append(tuned_pred.numpy(force=True))
            raw_predictions.append(raw_pred.numpy(force=True))

    df = pd.DataFrame()
    df[REFERENCE_KEY] = np.concatenate(references, axis=0)
    if translate_experiment_ids:
        int_ids = np.concatenate(experiments, axis=0)
        df[EXPID_KEY] = [resolve_dataspec(i).name for i in int_ids]
    else:
        df[EXPID_KEY] = np.concatenate(experiments, axis=0)
    df[TUNED_PRED_KEY] = np.concatenate(tuned_predictions, axis=0)
    df[RAW_PRED_KEY] = np.concatenate(raw_predictions, axis=0)

    if linear_calibration:
        df[TUNED_CALIBRATED_KEY] = np.nan
        df[RAW_CALIBRATED_KEY] = np.nan
        for exp_id in df[EXPID_KEY].unique():
            mask = df[EXPID_KEY] == exp_id
            ref = df.loc[mask, REFERENCE_KEY].values
            for pred_key, cal_key in [
                (TUNED_PRED_KEY, TUNED_CALIBRATED_KEY),
                (RAW_PRED_KEY, RAW_CALIBRATED_KEY),
            ]:
                pred = df.loc[mask, pred_key].values
                reg = LinearRegression()
                reg.fit(pred.reshape(-1, 1), ref)
                df.loc[mask, cal_key] = reg.predict(pred.reshape(-1, 1))

    # Add sequence information if dataset is a SequenceDataset
    if isinstance(dataset, SequenceDataset):
        sequences = list(dataset.sequences)
        df[SEQUENCE_KEY] = sequences
        df[MUTCOUNT_KEY] = _compute_mutation_distances(sequences, SARS_COV2_SEQ)

    return df


def report_dataset(
    dataset: SequenceDataset,  # type: ignore[type-arg]
    output: Union[str, Path],
    translate_experiment_ids: bool = True,
    include_reference: bool = True,
    batch_size: int = 256,
) -> None:
    """Write a CSV containing the features, sequences, and experiment IDs from a dataset.

    Arguments:
    ---------
    dataset:
        A SequenceDataset whose underlying dataset returns (features, signal, exp_id)
        tuples. Multi-dimensional feature tensors are flattened into 1-D vectors.
    output:
        Path to the CSV file to write.
    translate_experiment_ids:
        If True, integer labels of heads are translated to the string names of
        experiments. If False, raw integer IDs are used.
    include_reference:
        If True (default), include the reference signal value in the CSV under the
        column named by REFERENCE_KEY.
    batch_size:
        Batch size to use when iterating over the dataset.

    """
    loader = DataLoader(dataset, batch_size=batch_size)

    all_features = []
    all_experiments = []
    all_references = []

    for batch in loader:
        inp, signal, dataset_index = batch
        feat_np = inp.numpy(force=True)

        # Flatten non-batch dimensions into a single feature vector
        all_features.append(feat_np.reshape(feat_np.shape[0], -1))
        all_experiments.append(dataset_index.numpy(force=True))
        if include_reference:
            all_references.append(signal.numpy(force=True))

    features = np.concatenate(all_features, axis=0)
    experiments = np.concatenate(all_experiments, axis=0)

    df = pd.DataFrame(
        features,
        columns=pd.Index([f"feature_{i}" for i in range(features.shape[1])]),
    )

    if translate_experiment_ids:
        df[EXPID_KEY] = [resolve_dataspec(i).name for i in experiments]
    else:
        df[EXPID_KEY] = experiments

    df[SEQUENCE_KEY] = list(dataset.sequences)

    if include_reference:
        df[REFERENCE_KEY] = np.concatenate(all_references, axis=0)

    df.to_csv(output, index=False)

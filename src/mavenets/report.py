"""Routines for creating predictions using trained models."""
from typing import TypeVar, Final
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as pygDataLoader  # type: ignore
import pandas as pd  # type: ignore
from .network import MHTuner
from .tools import SIGNAL_PYGBATCHKEY, EXP_PYGBATCHKEY

_T = TypeVar("_T")

REFERENCE_KEY: Final = "reference"
TUNED_PRED_KEY: Final = "tuned"
RAW_PRED_KEY: Final = "raw"
EXPID_KEY: Final = "experiment"


def predict(
    model: MHTuner, dataset: Dataset, graph: bool = False, batch_size: int = 256
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
        as those used for training.
    graph:
        If model operates on pyg-style batches, this must be set to True.
    batch_size:
        Batch size to use when evaluating the predictions.

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
    df[EXPID_KEY] = np.concatenate(experiments, axis=0)
    df[TUNED_PRED_KEY] = np.concatenate(tuned_predictions, axis=0)
    df[RAW_PRED_KEY] = np.concatenate(raw_predictions, axis=0)
    return df

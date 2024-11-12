"""Provides routines for completing typical training tasks."""

from typing import Tuple, Callable, Any, Optional, Protocol, Union, Final
from warnings import warn
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader  # type: ignore
from tqdm import tqdm  # type: ignore
import pandas as pd  # type: ignore
from .util import null_contextmanager, sequence_mean, Patience
from .network import MHTuner

SIGNAL_PYGBATCHKEY: Final = "y"
EXP_PYGBATCHKEY: Final = "experiment"


class DoubleParameterizedLoss(Protocol):
    """Protocol for specialized loss function.

    As input, takes a 2-tuple of tensors for the signal, a single tensor for the
    reference, and an additional tensor to control the output.
    """

    def __call__(
        self,
        guesses: Tuple[torch.Tensor, torch.Tensor],
        reference: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate loss."""
        ...


class ParamTrainEval(Protocol):
    """Protocol for callable that minimizes a model.

    As input, takes model input, reference value, and a tensor passed to the
    loss function controlling its behavior.
    """

    def __call__(
        self,
        inp: Any,
        reference: torch.Tensor,
        loss_param: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate loss and adjust model."""
        ...


def mixed_MSE(
    guesses: Tuple[torch.Tensor, torch.Tensor],
    reference: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    """Calculate MSE loss on two inputs to a single reference and linearly combine them.

    Allows for training with an "annealing" schedule that switches the loss balance.

    Arguments:
    ---------
    guesses:
        2-Tuple, each element is a prediction.
    reference:
        Reference both inputs are compared against.
    mix:
        MSE losses are combined via (1.0-mix) and mix.

    Returns:
    -------
    Tensor with mean MSE.

    """
    first_mse = torch.nn.functional.mse_loss(guesses[0], reference)
    second_mse = torch.nn.functional.mse_loss(guesses[1], reference)
    return (1.0 - mix) * first_mse + mix * second_mse


def _create_parameterized_train_stepper(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[..., torch.Tensor],
    device: str,
    bfloat16: bool = False,
    compile: bool = False,
    compile_mode: Optional[str] = None,
    grad_clip: float = 1e3,
) -> ParamTrainEval:
    """Create function that can be called to perform an optimization step.

    Keyword only arguments.

    Arguments:
    ---------
    model:
        torch.nn.Module to optimize.
    optimizer:
        torch.optim optimizer (must already have been initialized)
    loss_function:
        Loss function that must take the entire model output as the first argument,
        reference as the second argument, and a third argument modulating its behavior.
    device:
        string specifying which device to use in wrapped function. Must be the same
        as the device of tensors layter supplied to returned function.
    bfloat16:
        If true, bf16 is used for part of the forward pass.
    compile:
        If true, compile the returned function.
    compile_mode:
        Passed to torch.compile if compile is True.
    grad_clip:
        Gradient norm to clip at.

    Returns:
    -------
    Function that can be called to

    """
    if bfloat16:
        precision_manager: Any = torch.autocast
    else:
        precision_manager = null_contextmanager

    def _train_step(
        inp: Any,
        reference: torch.Tensor,
        loss_param: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.train()
        with precision_manager(device_type=device, dtype=torch.bfloat16):
            pred = model(inp)
            loss_value = loss_function(pred, reference, loss_param)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        clip_grad_norm_(model.parameters(), grad_clip)
        return pred

    if compile:
        to_return = torch.compile(_train_step, mode=compile_mode)
    else:
        to_return = _train_step
    return to_return


class _MHTunerWrapper(torch.nn.Module):
    """Changes function signatures for smooth usage of compilation functions."""

    _GRAPH_TUNE_KEY: Final = EXP_PYGBATCHKEY

    def __init__(self, tuned_model: MHTuner, graph: bool = False) -> None:
        super().__init__()
        self.tuned_model = tuned_model
        self.graph = graph

    def forward(
        self,
        inp: Any,
    ) -> Tuple[Any, torch.Tensor]:
        if self.graph:
            return self.tuned_model(inp, inp[self._GRAPH_TUNE_KEY], return_raw=True)
        else:
            return self.tuned_model(inp[0], inp[1], return_raw=True)


def _create_parameterized_evaler(
    *,
    model: torch.nn.Module,
    loss_function: Callable[..., torch.Tensor],
    device: str,
    bfloat16: bool = False,
    compile: bool = False,
    compile_mode: Optional[str] = None,
) -> ParamTrainEval:
    """Create function that can be called to evaluate a nn.Module on to obtain a loss.

    Keyword only arguments.

    Similar to create_parameterized_train_stepper, but model is evaluated in eval mode
    and no optimization is performed. Useful to back in autocasting and pass compilation
    arguments.

    Arguments:
    ---------
    model:
        nn.Module to evaluate.
    loss_function:
        Loss function that must take the entire model output as the first argument,
        reference as the second argument, and a third argument modulating its behavior.
    device:
        string specifying which device to use in wrapped function. Must be the same
        as the device of tensors layter supplied to returned function.
    bfloat16:
        If true, bf16 is used for part of the forward pass.
    compile:
        If true, compile the returned function.
    compile_mode:
        If compile is True, passed to torch.compile.

    Returns:
    -------
    Callable that can map input and reference tensors, along with a loss
    parameter, to a loss value.

    """
    if bfloat16:
        precision_manager: Any = torch.autocast
    else:
        precision_manager = null_contextmanager

    def _eval(
        inp: Any,
        reference: torch.Tensor,
        loss_param: torch.Tensor,
    ) -> torch.Tensor:
        model.eval()
        with precision_manager(device_type=device, dtype=torch.bfloat16):
            pred = model(inp)
            loss_value = loss_function(pred, reference, loss_param)
        return loss_value

    if compile:
        to_return = torch.compile(_eval, mode=compile_mode)
    else:
        to_return = _eval
    return to_return


def _eval_dataset(
    evaler: ParamTrainEval,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    loss_param: torch.Tensor,
    graph: bool = False,
) -> float:
    """Load and evaluate loss on given data."""
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
    losses = []
    for batch in loader:
        if graph:
            full_inp = batch
            signal = batch[SIGNAL_PYGBATCHKEY]
            dataset_index = batch[EXP_PYGBATCHKEY]
        else:
            inp, signal, dataset_index = batch
            full_inp = (inp, dataset_index)
        loss = evaler(
            inp=full_inp,
            reference=signal,
            loss_param=loss_param,
        ).item()
        losses.append(loss)
    return sequence_mean(losses)


def train_tunable_model(
    model: MHTuner,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    train_loss_function: DoubleParameterizedLoss = mixed_MSE,
    report_loss_function: DoubleParameterizedLoss = mixed_MSE,
    train_batch_size: int = 32,
    reporting_batch_size: int = 512,
    report_stride: int = 5,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    train_bfloat16: bool = True,
    start_loss_param: Union[float, torch.Tensor] = 0.5,
    end_loss_param: Union[float, torch.Tensor] = 0.0,
    loss_param_ramp_size: int = 20,
    grad_clip: float = 1e3,
    progress_bar: bool = False,
    patience: int = 50,
    graph: bool = False,
) -> Tuple[int, float, pd.DataFrame]:
    """Train MHTuner instance.

    Arguments:
    ---------
    model:
        Model that will be optimized. Must be a MHTuner instance, not just a nn.Module.
    optimizer:
        torch.optim optimizer instance. Must already be initialized.
    device:
        Device to train on; likely "cpu" or "cuda".
    n_epochs:
        Maximum number of epochs to train for. If no improvement in the validation loss
        is observed over a given time frame (see patience argument), training
        may be halted before reading this number of epochs.
    train_dataset:
        Validation dataset. If graph is True, should be serve pyg Data objects with
        x, y, and experiment attributes. Else, should return
        (features, signal, experiment) tuples.
    valid_dataset:
        Validation dataset. If graph is True, should be serve pyg Data objects with
        x, y, and experiment attributes. Else, should return
        (features, signal, experiment) tuples.
    train_loss_function:
        Train loss function to use. Must be a parameterized loss: should map
        three arguments (guess, reference, parameter) to a scalar tensor.
        This is likely an instance of mixed_MSE.
    report_loss_function:
        Train loss function to use. Must be a parameterized loss: should map
        three arguments (guess, reference, parameter) to a scalar tensor.
        This is likely an instance of mixed_MSE.
    train_batch_size:
        Batch size for training.
    reporting_batch_size:
        Batch size for reporting losses.
    report_stride:
        Loss reporting is done at intevals of this size. E.g., if 5, we report losses
        every 5 epochs. Larger values result in faster training, but less information.
    compile:
        Whether to compile the train and eval calls.
    compile_mode:
        Passed to torch.compile if compile is True.
    train_bfloat16:
        Whether to run forward pass in bfloat16 precision. May not be supported on all
        GPUs.
    start_loss_param:
        Optimization is performed using a loss change changes during training by passing
        a control parameter. This parameter starts at start_loss_param and linearly
        ramps down to end_loss_param over loss_param_ramp_size epochs; after this ramp,
        the parameter stays at end_loss for the remainder of training.
    end_loss_param:
        See start_loss_param.
    loss_param_ramp_size:
        See start_loss_param.
    grad_clip:
        Grad norm to clip at. Lower this is training all of a sudden becomes unstable.
    progress_bar:
        Whether to show a tqdm progress bar updating every epoch.
    patience:
        Training is terminated one the (non-smoothed) validation loss has not improved
        for this many updates (not epochs--- epochs*report_stride).
    graph:
        Treats model as a torch geometric graph neural network. Effectively, if
        model take torch_geometric Data objects as input, this must be True. Changes
        how models are wrapped and objects used for data loading.

    Returns:
    -------
    3-Tuple: First element is the epoch where the optimal validation loss is found
    after smoothing data with 3-item rolling median, second element is the smoothed
    loss at that epoch, and third is a pd.DataFrame containing logged training and
    validation losses.

    """
    loss_param_schedule = list(
        torch.linspace(
            start_loss_param, end_loss_param, loss_param_ramp_size, device=device
        )
    )

    wrapped_model = _MHTunerWrapper(model, graph=graph)

    train_stepper = _create_parameterized_train_stepper(
        model=wrapped_model,
        optimizer=optimizer,
        device=device,
        bfloat16=train_bfloat16,
        loss_function=train_loss_function,
        compile=compile,
        compile_mode=compile_mode,
        grad_clip=grad_clip,
    )

    evaler = _create_parameterized_evaler(
        model=wrapped_model,
        device=device,
        loss_function=report_loss_function,
        compile=compile,
        compile_mode=compile_mode,
    )

    if graph:
        train_dataloader = pygDataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True
        )

    patience_counter: Patience[float] = Patience()

    records = []
    record_epochs = []
    if progress_bar:
        epoch_source = tqdm(range(n_epochs))
    else:
        epoch_source = range(n_epochs)
    for epoch in epoch_source:
        try:
            loss_param = loss_param_schedule.pop(0)
        except IndexError:
            pass
        for batch in train_dataloader:
            if graph:
                full_inp = batch
                signal = batch["y"]
            else:
                inp, signal, dataset_index = batch
                full_inp = (inp, dataset_index)
            train_stepper(inp=full_inp, reference=signal, loss_param=loss_param)

        if epoch % report_stride == 0:
            with torch.no_grad():
                epoch_train_loss = _eval_dataset(
                    evaler=evaler,
                    dataset=train_dataset,
                    batch_size=reporting_batch_size,
                    loss_param=loss_param,
                    graph=graph,
                )
                epoch_val_loss = _eval_dataset(
                    evaler=evaler,
                    dataset=valid_dataset,
                    batch_size=reporting_batch_size,
                    loss_param=loss_param,
                    graph=graph,
                )
            record_epochs.append(epoch)
            records.append((epoch_train_loss, epoch_val_loss, loss_param.item()))
            patience_count = patience_counter.consider(epoch_val_loss)
            if patience_count > patience / report_stride:
                break
            if progress_bar:
                epoch_source.set_description(
                    "{}/{}".format(round(epoch_val_loss, 3), patience_count)
                )
    else:
        warn(
            "Maximum number of epochs reached. Possibly not converged.",
            UserWarning,
            stacklevel=1,
        )

    table = pd.DataFrame(
        records, record_epochs, columns=["train_loss", "valid_loss", "loss_param"]
    )
    rolled_vals = table["valid_loss"].rolling(3, center=True).median()
    best_epoch = rolled_vals.index[rolled_vals.argmin(skipna=True)]
    best_val = rolled_vals.min()
    return best_epoch, best_val, table

"""Provides routines for completing typical training tasks."""

from typing import Tuple, Callable, Any, Optional, Protocol
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from .util import null_contextmanager, sequence_mean
from .network import MHTuner

# was this: loss = cal_loss + mix * raw_loss
# for later: loss = mix_cal * cal_loss + mix_raw * raw_loss


class DoubleParameterizedLoss(Protocol):
    """Specialized loss function.

    As input, takes a 2-tupel of tensors for the signal, a single tensor for the
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
    """Callable that minimizes a model.

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
    first_mse = torch.nn.functional.mse_loss(guesses[0], reference)
    second_mse = torch.nn.functional.mse_loss(guesses[1], reference)
    return (1.0 - mix) * first_mse + mix * second_mse


def create_parameterized_train_stepper(
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


def create_evaler(
    *,
    model: torch.nn.Module,
    loss_function: Callable[..., torch.Tensor],
    device: str,
    bfloat16: bool = False,
    compile: bool = False,
    compile_mode: Optional[str] = None,
) -> ParamTrainEval:

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


class _MHTunerWrapper(torch.nn.Module):
    """Changes function signatures for smooth usage of compilation functions."""

    def __init__(self, tuned_model: MHTuner) -> None:
        super().__init__()
        self.tuned_model = tuned_model

    def forward(
        self, inp: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tuned_model(inp[0], inp[1], return_raw=True)


def train_tunable_model(
    model: MHTuner,
    optimizer: torch.optim.Optimizer,
    device: str,
    n_epochs: int,
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    loss_function: DoubleParameterizedLoss = mixed_MSE,
    train_batch_size: int = 32,
    reporting_batch_size: int = 512,
    report_stride: int = 5,
    compile: bool = False,
    compile_mode: str = "reduce-overhead",
    train_bfloat16: bool = True,
    loss_param: Optional[torch.Tensor] = None,
    grad_clip: float = 1e3,
) -> None:
    if loss_param is None:
        loss_param = torch.tensor(0.1).to(device)

    wrapped_model = _MHTunerWrapper(model)

    train_stepper = create_parameterized_train_stepper(
        model=wrapped_model,
        optimizer=optimizer,
        device=device,
        bfloat16=train_bfloat16,
        loss_function=loss_function,
        compile=compile,
        compile_mode=compile_mode,
        grad_clip=grad_clip,
    )

    evaler = create_evaler(
        model=wrapped_model,
        device=device,
        loss_function=loss_function,
        compile=compile,
        compile_mode=compile_mode,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    reporting_valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=reporting_batch_size,
    )
    reporting_train_dataloader = DataLoader(
        train_dataset,
        batch_size=reporting_batch_size,
    )
    for epoch in range(n_epochs):
        for seq, signal, dataset_index in train_dataloader:
            train_stepper(
                inp=(seq, dataset_index), reference=signal, loss_param=loss_param
            )
        if epoch % report_stride == 0:
            with torch.no_grad():
                epoch_train_loss = sequence_mean(
                    [
                        evaler(
                            inp=(seq, dataset_index),
                            reference=signal,
                            loss_param=loss_param,
                        ).item()
                        for seq, signal, dataset_index in reporting_train_dataloader
                    ]
                )
                epoch_val_loss = sequence_mean(
                    [
                        evaler(
                            inp=(seq, dataset_index),
                            reference=signal,
                            loss_param=loss_param,
                        ).item()
                        for seq, signal, dataset_index in reporting_valid_dataloader
                    ]
                )
                print(epoch_train_loss, epoch_val_loss)

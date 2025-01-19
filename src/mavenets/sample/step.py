"""Tools for Metropolis Monte Carlo simulations."""

from typing import Callable, Optional, Sequence, List, Union
from dataclasses import dataclass
import torch
from ..data import get_default_int_encoder, SARS_COV2_SEQ
from ..util import num_changes


@dataclass(frozen=True)
class State:
    """Describes the state of a simulation."""

    index: int
    sequence: torch.Tensor  # int32
    energy: Optional[float] = None


class IntMutate:
    """Mutates an integer sequence with a integer drawn from a uniform distribution.

    Performs transformations like [ 1, 2, 3 ] -> [ 1, 4, 3 ].
                                       ^              ^
    Only one mutation is applied each time. Multiple instances of this mutation starting
    from a single tarting sequence are performed.
    """

    def __init__(
        self,
        min_int: int,
        max_int: int,
        n_mutations: int = 1,
    ) -> None:
        """Store options.

        Arguments:
        ---------
        min_int:
            Minimum integer (i.e., inclusive) to use as a mutation outcome.
        max_int:
            One larger (i.e., exclusive) that the maximum integer to use as a mutation
            outcome.
        n_mutations:
            Number of positions to change in each mutation. Currently, only 1 is
            supported. Note that since multiple mutation candidates are returned, more
            than one mutation will (likely )be obtained in aggregate; however, each
            candidate sequence will mutated in a maximum of 1 times.

        """
        if n_mutations != 1:
            raise NotImplementedError("Only single mutations are supported.")
        self.n_mutations = n_mutations
        self.min_int = min_int
        self.max_int = max_int

    def __call__(self, start: torch.Tensor, n_mutants: int) -> torch.Tensor:
        """Create mutated version of input.

        Arguments:
        ---------
        start:
            Sequence to mutate. Tensor.
        n_mutants:
            Number of mutants to suggest.

        Returns:
        -------
        Tensor of shape (n_mutants, size) where size is the size of start.

        """
        # copy the input sequence multiple times.
        candidates = torch.tile(start, dims=(n_mutants, 1))
        # create all possible mutations in a second array
        mutations = torch.randint(self.min_int, self.max_int, size=(n_mutants,)).to(
            start
        )
        # randomly select where those mutations will be applied
        positions = torch.randint(0, start.shape[0], size=(n_mutants,)).to(start)
        # modify the start copies by copying in the generated mutations.
        candidates[torch.arange(n_mutants), positions] = mutations
        return candidates


class BiasedIntMutate:
    """Mutates an integer sequence using a uniform distribution and reference sequence.

    Performs transformations like [ 1, 2, 3 ] -> [ 1, 4, 3 ].
                                       ^              ^
    Only one mutation is applied each time. Multiple instances of this mutation starting
    from a single tarting sequence are performed.

    When mutating, first a position along the sequence is elected. Then, we
    randomly decide whether to use a fallback (center) residue that was
    supplied at initialization or a random mutation. If using the fallback, we
    use the residue at the chosen position in the center sequence. If a random
    mutation, we select a random integer from a uniform distribution.

    Warning:
    -------
    Note that this is an asymmetric proposal distribution, and may modify the stationary
    distribution of simulation classes in this module.

    """

    def __init__(
        self,
        min_int: int,
        max_int: int,
        bias: float,  # 1 is only native, 0 is no bias
        center: torch.Tensor,
        n_mutations: int = 1,
    ) -> None:
        """Store options.

        Arguments:
        ---------
        min_int:
            Minimum integer (i.e., inclusive) to use as a mutation outcome.
        max_int:
            One larger (i.e., exclusive) that the maximum integer to use as a mutation
            outcome.
        n_mutations:
            Number of positions to change in each mutation. Currently, only 1 is
            supported.
        bias:
            Float between 0 and 1 characterize how often to fall back to center. If one,
            we always propose center's corresponding integer. If 0, we never propose
            center's integer.
        center:
            Sequence to fall back to.

        """
        if n_mutations != 1:
            raise NotImplementedError("Only single mutations are supported.")
        self.n_mutations = n_mutations
        self.min_int = min_int
        self.max_int = max_int
        self.bias = bias
        self.center = center

    def __call__(self, start: torch.Tensor, n_mutants: int) -> torch.Tensor:
        """Create mutated version of input.

        Arguments:
        ---------
        start:
            Sequence to mutate. Tensor.
        n_mutants:
            Number of mutants to suggest.

        Returns:
        -------
        Tensor of shape (n_mutants, size) where size is the size of start.

        """
        # copy the input sequence multiple times.
        candidates = torch.tile(start, dims=(n_mutants, 1))
        # create all possible random mutations in a second array
        mutations = torch.randint(self.min_int, self.max_int, size=(n_mutants,)).to(
            start
        )
        # randomly select where those mutations will be applied
        positions = torch.randint(0, start.shape[0], size=(n_mutants,)).to(start)
        # given those positions, get what they would correspond to in the center
        fallbacks = self.center[positions]
        # randomly select one of the mutations or the center items
        changes = torch.where(
            torch.rand(size=(n_mutants,), device=fallbacks.device) < self.bias,
            fallbacks,
            mutations,
        )
        # apply changes to copied input sequence
        candidates[torch.arange(n_mutants), positions] = changes
        return candidates


def _metropolis_crit(
    deltas: torch.Tensor, beta: Union[torch.Tensor, float]
) -> torch.Tensor:
    """Apply metropolis criteria.

    Applies metroplis criteria given an array of energy differences. The
    targeted density is assumed to be proportional to exp(-beta U(.)); deltas
    contains deltas: = U(proposed) - U(starting). We return
    `min(1, p(proposed)/p(start))`.

    Arguments:
    ---------
    deltas:
        Difference in energy between proposed and starting sequence.
    beta:
        Beta of the targeted distribution.

    Returns:
    -------
    Torch tensor of calculated criteria.

    """
    vals = torch.exp(-beta * deltas)
    return torch.clip(vals, max=1.0)


class MetStep:
    """Performs a batched Metropolis update.

    May correspond to more than a single formal accept/reject step.

    Given an input sequence, a proposer object is used to create multiple move
    candidates.  These candidates, and the input sequence, are fed into the
    model to predict their energy. The Metropolis criteria is evaluated on all
    of the candidates,  and the first candidate giving a stochastic accept is
    isolated as the returned sequence.

    The index of the returned sequence is the index of the input sequence plus
    the number of implied steps. For example, if 5 candidates are being evaluated
    and acceptance decisions look like this: [False, True, False, True, True], the
    second candidate is returned with an implied decision increment of 2 (since
    we rejected the first candidate). If no acceptance decision is generated,
    the input sequence is returned with a suitably incremented index.

    Note that this class does NOT calculate a Metropolis-Hastings step, only a
    Metropolis step. Using an asymmetric proposal distribution will modify
    the stationary distribution of the Markov Chain.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        proposer: Callable[[torch.Tensor, int], torch.Tensor],
        batch_size: int = 512,
        beta: float = 1.0,
        center: Optional[torch.Tensor] = None,
        max_distance_to_center: Optional[int] = None,
        compile: bool = False,
    ) -> None:
        """Store options.

        Arguments:
        ---------
        model:
            Callable (likely a torch.nn.Module) that maps Tensors to Tensors. Treated
            as the energy in the definition of a Boltzmann distribution.
        proposer:
            Callable that takes a sequence of shape (size,) and returns
            candidates of shape (n_candidates, size) to evaluate the Metropolis criteria
            on.  Likely IntMutate or BiasedIntMutate.
        batch_size:
            Number of candidates to evaluate when taking steps.
        beta:
            Probability density is assumed to proportional to exp(-beta U), where
            U is the output of model.
        center:
            If center is provided, proposals which would move the system further than
            max_distance_to_center are rejected. Both or neither center and
            max_distance_to_center may be not-none. Distance is the number of shared
            elements, not euclidean distance.
        max_distance_to_center:
            see center argument.
        compile:
            Whether to use torch.compile to speed up computation. Usually improves
            performance, but may make debugging harder.

        """
        self.model = model
        self.proposer = proposer
        self.batch_size = batch_size
        self._beta = beta
        self.beta: Optional[torch.Tensor] = None
        self.compile = compile
        self._compiled_step: Optional[Callable] = None

        if center is not None and max_distance_to_center is None:
            raise ValueError(
                "center sequence was provided but max_distance_to_center "
                "was unset. Either provide both or neither."
            )
        if center is None:
            self.bcast_center = None
        else:
            self.bcast_center = torch.unsqueeze(center, 0)
        self.max_distance_to_center = max_distance_to_center

    def _step(self, inp: State) -> State:
        """Return new simulation state."""
        start = inp.sequence
        if self.beta is None:
            self.beta = torch.tensor(self._beta, device=start.device)
        # get candidates
        cands = self.proposer(start, self.batch_size)
        # get energy of starting sequence
        start_e = self.model(start[None, :])
        # get energy of candidates
        next_es = self.model(cands)
        # get metropolis crits
        deltas = next_es - start_e
        crits = _metropolis_crit(deltas, self.beta)
        # do random check for acceptance
        variates = torch.rand(*crits.shape, device=start.device)
        # whether each sample would be an accept based on energy
        acceptances = variates < crits
        # if any acceptances would be to far in distance, reject them
        if self.bcast_center is not None:
            # get which samples would be far
            mask = num_changes(self.bcast_center, cands) <= self.max_distance_to_center
            # boolean and
            acceptances = acceptances * mask

        # see if any samples were accepted
        which_sample = torch.nonzero(acceptances, as_tuple=True)[0]
        # if no sample was accepted, return original sample with incremented index
        if which_sample.shape[0] == 0:
            new = State(
                index=inp.index + self.batch_size, sequence=start, energy=inp.energy
            )
        # else return new state using first accepted sample
        else:
            new = State(
                index=int(inp.index + which_sample[0] + 1),
                sequence=cands[which_sample[0]].clone(),
                energy=next_es[which_sample[0]].item(),
            )
        return new

    def step(self, inp: State) -> State:
        """Make a single Metropolis update.

        May correspond to multiple Metropolis updates in the implied
        Markov chain.
        """
        if self.compile:
            if self._compiled_step is None:
                self._compiled_step = torch.compile(self._step)
            return self._compiled_step(inp)
        else:
            return self._step(inp)

    def __call__(self, inp: State) -> State:
        """Make a single Metropolis update.

        May correspond to multiple Metropolis updates in the implied
        Markov chain.
        """
        return self.step(inp)


class MetSim:
    """Perform a simulation composed of Metropolis moves.

    Distribution is assumed to be of the form `exp(- beta U(.))`, where
    `U` is a user-supplied callable.  Repeatedly applies MetStep.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        proposer: Callable[[torch.Tensor, int], torch.Tensor],
        batch_size: int = 128,
        beta: float = 1.0,
        center: Optional[torch.Tensor] = None,
        max_distance_to_center: Optional[int] = None,
        compile: bool = False,
        jump_stride: int = 10,
    ) -> None:
        """Initialze stepper and store options.

        Arguments:
        ---------
        model:
            Callable that represents the energy of a given sequence.
        proposer:
            Callable that takes a sequence of shape (size,) and returns
            candidates of shape (n_candidates, size) to evaluate the Metropolis criteria
            on.  Likely IntMutate or BiasedIntMutate.
        batch_size:
            Number of candidates to evaluate when taking steps.
        beta:
            Probability density is assumed to proportional to exp(-beta U), where
            U is the output of model.
        center:
            Passed to MetStep.  If center is provided, proposals which would
            move the system further than max_distance_to_center are rejected.
            Both or neither center and max_distance_to_center may be not-none.
            Distance is the number of shared elements, not euclidean distance.
        max_distance_to_center:
            see center argument.
        compile:
            Whether to use torch.compile to speed up computation. Usually improves
            performance, but may make debugging harder.
        jump_stride:
            Only a subset of jumps are recorded; this value controls how many.
            For example, 5 implies that the state every 5 jumps is recorded. Note that
            jumps are not the same as the underlying steps in the Markov chain; see
            MetStep for more information.

        """
        self.stepper = MetStep(
            model=model,
            proposer=proposer,
            batch_size=batch_size,
            beta=beta,
            center=center,
            max_distance_to_center=max_distance_to_center,
            compile=compile,
        )
        self.jump_stride = jump_stride

    def propagate(self, n_jumps: int, start: State) -> State:
        """Propagate the chain forward in time.

        Arguments:
        ---------
        n_jumps:
            Number of jumps to perform.
        start:
            State object to start jumps from.

        Returns:
        -------
        State instance.

        """
        frame = start
        for _ in range(n_jumps):
            frame = self.stepper(frame)
        return frame

    def run(
        self,
        n_steps: int,
        start: Optional[Sequence[int]] = None,
        device: Optional[str] = None,
    ) -> List[State]:
        """Run a simulation.

        Arguments:
        ---------
        n_steps:
            Approximate number of chain steps to run the simulation for: the final
            returned State will have index larger than this.
        start:
            Sequence to start simulation from. If None (default), the integer-encoded
            wild type SARS COV2 sequence is used.
        device:
            torch Device to default to when creating tensors. Likely "cpu" or "cuda".

        Returns:
        -------
        List of State instances.

        """
        with torch.no_grad():
            if start is None:
                start_tensor = (
                    get_default_int_encoder()
                    .encode(SARS_COV2_SEQ, tensor=True)
                    .to(device)
                )
            else:
                start_tensor = torch.tensor(start, device=device)
            frames = []
            state = State(index=0, sequence=start_tensor)
            frames.append(state)
            while True:
                state = self.propagate(self.jump_stride, state)
                frames.append(state)
                if state.index > n_steps:
                    break
        return frames

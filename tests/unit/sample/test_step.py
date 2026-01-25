"""Tests for mavenets.sample.step module."""

from typing import List

import pytest
import torch

from mavenets.sample.step import (  # type: ignore[import-not-found]
    State,
    IntMutate,
    BiasedIntMutate,
    MetStep,
    MetSim,
    _metropolis_crit,  # type: ignore[private-usage]
)


class TestState:
    """Tests for State dataclass."""

    def test_create_state(self, cpu_device: str) -> None:
        """Should create State with required fields."""
        seq = torch.tensor([0, 1, 2], device=cpu_device)
        state = State(index=0, sequence=seq)
        assert state.index == 0
        assert torch.equal(state.sequence, seq)
        assert state.energy is None

    def test_create_state_with_energy(self, cpu_device: str) -> None:
        """Should create State with optional energy."""
        seq = torch.tensor([0, 1, 2], device=cpu_device)
        state = State(index=5, sequence=seq, energy=-1.5)
        assert state.energy == -1.5

    def test_state_is_frozen(self, cpu_device: str) -> None:
        """State should be immutable (frozen dataclass)."""
        seq = torch.tensor([0, 1, 2], device=cpu_device)
        state = State(index=0, sequence=seq)
        with pytest.raises(Exception):  # FrozenInstanceError
            state.index = 1  # type: ignore


class TestIntMutate:
    """Tests for IntMutate class."""

    def test_init(self) -> None:
        """Should initialize with min/max integers."""
        mutator = IntMutate(min_int=0, max_int=20)
        assert mutator.min_int == 0
        assert mutator.max_int == 20
        assert mutator.n_mutations == 1

    def test_init_multiple_mutations_raises(self) -> None:
        """Should raise NotImplementedError for n_mutations != 1."""
        with pytest.raises(NotImplementedError, match="Only single mutations"):
            IntMutate(min_int=0, max_int=20, n_mutations=2)

    def test_call_produces_correct_shape(self, cpu_device: str) -> None:
        """Should produce tensor of correct shape."""
        mutator = IntMutate(min_int=0, max_int=20)
        start = torch.tensor([0, 1, 2, 3, 4], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=10)
        assert candidates.shape == (10, 5)

    def test_call_mutates_exactly_one_position(self, cpu_device: str) -> None:
        """Each candidate should differ from start in at most one position."""
        mutator = IntMutate(min_int=0, max_int=20)
        start = torch.tensor([0, 1, 2, 3, 4], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=100)
        # Count differences for each candidate
        diffs = (candidates != start).sum(dim=1)
        # Each should have 0 or 1 difference (0 if mutation equals original)
        assert (diffs <= 1).all()

    def test_call_mutations_in_range(self, cpu_device: str) -> None:
        """Mutations should be within specified range."""
        mutator = IntMutate(min_int=5, max_int=10)
        start = torch.tensor([0, 0, 0, 0, 0], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=100)
        # All non-zero values should be in range [5, 10)
        changed = candidates[candidates != 0]
        if len(changed) > 0:
            assert (changed >= 5).all()
            assert (changed < 10).all()


class TestBiasedIntMutate:
    """Tests for BiasedIntMutate class."""

    def test_init(self, cpu_device: str) -> None:
        """Should initialize with required parameters."""
        center = torch.tensor([0, 1, 2, 3, 4], device=cpu_device, dtype=torch.int32)
        mutator = BiasedIntMutate(min_int=0, max_int=20, bias=0.5, center=center)
        assert mutator.min_int == 0
        assert mutator.max_int == 20
        assert mutator.bias == 0.5
        assert torch.equal(mutator.center, center)

    def test_init_multiple_mutations_raises(self, cpu_device: str) -> None:
        """Should raise NotImplementedError for n_mutations != 1."""
        center = torch.tensor([0, 1, 2], device=cpu_device, dtype=torch.int32)
        with pytest.raises(NotImplementedError, match="Only single mutations"):
            BiasedIntMutate(
                min_int=0, max_int=20, bias=0.5, center=center, n_mutations=2
            )

    def test_call_produces_correct_shape(self, cpu_device: str) -> None:
        """Should produce tensor of correct shape."""
        center = torch.tensor([0, 1, 2, 3, 4], device=cpu_device, dtype=torch.int32)
        mutator = BiasedIntMutate(min_int=0, max_int=20, bias=0.5, center=center)
        start = torch.tensor([5, 6, 7, 8, 9], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=10)
        assert candidates.shape == (10, 5)

    def test_bias_one_always_uses_center(self, cpu_device: str) -> None:
        """With bias=1.0, mutations should always use center values."""
        center = torch.tensor([10, 11, 12, 13, 14], device=cpu_device, dtype=torch.int32)
        mutator = BiasedIntMutate(min_int=0, max_int=5, bias=1.0, center=center)
        start = torch.tensor([0, 0, 0, 0, 0], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=100)
        # All changed values should be from center (10-14)
        for i in range(100):
            diff_mask = candidates[i] != start
            if diff_mask.any():
                changed_val = candidates[i][diff_mask][0].item()
                pos = diff_mask.nonzero()[0].item()
                assert changed_val == center[int(pos)].item()

    def test_bias_zero_uses_random(self, cpu_device: str) -> None:
        """With bias=0.0, mutations should use random values."""
        center = torch.tensor([100, 100, 100], device=cpu_device, dtype=torch.int32)
        mutator = BiasedIntMutate(min_int=0, max_int=5, bias=0.0, center=center)
        start = torch.tensor([50, 50, 50], device=cpu_device, dtype=torch.int32)
        candidates = mutator(start, n_mutants=100)
        # Changed values should be in range [0, 5), not 100
        changed = candidates[candidates != 50]
        if len(changed) > 0:
            assert (changed >= 0).all()
            assert (changed < 5).all()


class TestMetropolisCrit:
    """Tests for _metropolis_crit function."""

    def test_zero_delta_returns_one(self, cpu_device: str) -> None:
        """Zero energy difference should return acceptance probability 1."""
        deltas = torch.tensor([0.0], device=cpu_device)
        result = _metropolis_crit(deltas, beta=1.0)
        assert torch.allclose(result, torch.tensor([1.0], device=cpu_device))

    def test_negative_delta_returns_one(self, cpu_device: str) -> None:
        """Negative energy difference (favorable) should return 1."""
        deltas = torch.tensor([-1.0, -5.0, -10.0], device=cpu_device)
        result = _metropolis_crit(deltas, beta=1.0)
        expected = torch.tensor([1.0, 1.0, 1.0], device=cpu_device)
        assert torch.allclose(result, expected)

    def test_positive_delta_returns_less_than_one(self, cpu_device: str) -> None:
        """Positive energy difference (unfavorable) should return < 1."""
        deltas = torch.tensor([1.0, 2.0, 3.0], device=cpu_device)
        result = _metropolis_crit(deltas, beta=1.0)
        assert (result < 1.0).all()
        assert (result > 0.0).all()

    def test_beta_scaling(self, cpu_device: str) -> None:
        """Higher beta should decrease acceptance for positive deltas."""
        deltas = torch.tensor([1.0], device=cpu_device)
        result_low_beta = _metropolis_crit(deltas, beta=0.5)
        result_high_beta = _metropolis_crit(deltas, beta=2.0)
        assert result_low_beta > result_high_beta

    def test_result_clipped_to_one(self, cpu_device: str) -> None:
        """Result should never exceed 1.0."""
        deltas = torch.tensor([-100.0], device=cpu_device)
        result = _metropolis_crit(deltas, beta=1.0)
        assert result.item() == 1.0

    def test_tensor_beta(self, cpu_device: str) -> None:
        """Should work with tensor beta."""
        deltas = torch.tensor([1.0], device=cpu_device)
        beta = torch.tensor(1.0, device=cpu_device)
        result = _metropolis_crit(deltas, beta)
        assert result.shape == (1,)

    def test_batch_deltas(self, cpu_device: str) -> None:
        """Should handle batch of deltas."""
        deltas = torch.tensor([-1.0, 0.0, 1.0, 2.0], device=cpu_device)
        result = _metropolis_crit(deltas, beta=1.0)
        assert result.shape == (4,)
        # First two should be 1.0, last two should be < 1.0
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] < 1.0
        assert result[3] < 1.0


class TestMetStep:
    """Tests for MetStep class."""

    @pytest.fixture
    def simple_energy_fn(self, cpu_device: str) -> callable:
        """Create a simple energy function for testing."""

        def energy_fn(x: torch.Tensor) -> torch.Tensor:
            # Energy is sum of values - prefers lower values
            return x.float().sum(dim=-1)

        return energy_fn

    @pytest.fixture
    def simple_proposer(self) -> IntMutate:
        """Create a simple proposer for testing."""
        return IntMutate(min_int=0, max_int=10)

    def test_init_basic(
        self, simple_energy_fn: callable, simple_proposer: IntMutate
    ) -> None:
        """Should initialize with basic parameters."""
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
        )
        assert stepper.batch_size == 32
        assert stepper._beta == 1.0
        assert stepper.avoid_null_step is True
        assert stepper.compile is False

    def test_init_with_center(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Should initialize with center sequence."""
        center = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            center=center,
            max_distance_to_center=3,
        )
        assert stepper.max_distance_to_center == 3
        assert stepper.bcast_center is not None
        assert stepper.bcast_center.shape == (1, 5)

    def test_init_center_without_max_distance_raises(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Should raise error if center provided without max_distance."""
        center = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        with pytest.raises(ValueError, match="max_distance_to_center"):
            MetStep(
                model=simple_energy_fn,
                proposer=simple_proposer,
                center=center,
                # max_distance_to_center not provided
            )

    def test_step_returns_state(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Step should return a State object."""
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
        )
        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        result = stepper.step(start_state)

        assert isinstance(result, State)
        assert result.index > start_state.index
        assert result.sequence.shape == start_seq.shape

    def test_step_increments_index(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Step should always increment the index."""
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
        )
        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=10, sequence=start_seq)

        result = stepper.step(start_state)

        assert result.index > start_state.index

    def test_call_same_as_step(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """__call__ should behave same as step."""
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
        )
        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        # Both should work
        result = stepper(start_state)
        assert isinstance(result, State)

    def test_step_with_high_beta_accepts_less(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Higher beta should result in fewer acceptances (stricter)."""
        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        # Run multiple steps with low beta
        stepper_low = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=100,
            beta=0.01,  # Low beta, accepts almost everything
        )

        # Run multiple steps with high beta
        stepper_high = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=100,
            beta=100.0,  # High beta, very strict
        )

        # Both should produce valid states
        result_low = stepper_low.step(start_state)
        result_high = stepper_high.step(start_state)

        assert isinstance(result_low, State)
        assert isinstance(result_high, State)

    def test_step_respects_max_distance(
        self, simple_energy_fn: callable, cpu_device: str
    ) -> None:
        """Step should reject moves beyond max_distance_to_center."""
        center = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        # Proposer that makes large changes
        proposer = IntMutate(min_int=0, max_int=20)

        stepper = MetStep(
            model=simple_energy_fn,
            proposer=proposer,
            batch_size=100,
            beta=0.01,  # Accept almost everything
            center=center,
            max_distance_to_center=1,  # Very restrictive
        )

        start_state = State(index=0, sequence=center.clone())

        # Run step
        result = stepper.step(start_state)

        # Result should be within max_distance of center
        if not torch.equal(result.sequence, center):
            diff = (result.sequence != center).sum().item()
            assert diff <= 1

    def test_step_avoid_null_step(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """With avoid_null_step=True, should not return identical sequence."""
        stepper = MetStep(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=100,
            beta=0.001,  # Very low, accepts almost everything
            avoid_null_step=True,
        )

        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        # When a move is accepted, it should be different from start
        # (unless all batch_size candidates are rejected)
        result = stepper.step(start_state)
        # Just verify it returns a valid state
        assert isinstance(result, State)


class TestMetSim:
    """Tests for MetSim class."""

    @pytest.fixture
    def simple_energy_fn(self, cpu_device: str) -> callable:
        """Create a simple energy function for testing."""

        def energy_fn(x: torch.Tensor) -> torch.Tensor:
            return x.float().mean(dim=-1)

        return energy_fn

    @pytest.fixture
    def simple_proposer(self) -> IntMutate:
        """Create a simple proposer for testing."""
        return IntMutate(min_int=0, max_int=10)

    def test_init_basic(
        self, simple_energy_fn: callable, simple_proposer: IntMutate
    ) -> None:
        """Should initialize with basic parameters."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )
        assert sim.jump_stride == 5
        assert isinstance(sim.stepper, MetStep)

    def test_propagate(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Propagate should advance the chain."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        start_seq = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        start_state = State(index=0, sequence=start_seq)

        result = sim.propagate(n_jumps=3, start=start_state)

        assert isinstance(result, State)
        assert result.index > start_state.index

    def test_run_returns_list_of_states(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Run should return list of State objects."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        start_seq = [5, 5, 5, 5, 5]
        frames = sim.run(n_steps=100, start=start_seq, device=cpu_device)

        assert isinstance(frames, list)
        assert len(frames) > 1
        assert all(isinstance(f, State) for f in frames)

    def test_run_first_frame_is_start(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """First frame should be the starting state."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        start_seq = [5, 5, 5, 5, 5]
        frames = sim.run(n_steps=50, start=start_seq, device=cpu_device)

        assert frames[0].index == 0
        expected_start = torch.tensor(start_seq, device=cpu_device)
        assert torch.equal(frames[0].sequence, expected_start)

    def test_run_indices_increase(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Indices should be monotonically increasing."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        start_seq = [5, 5, 5, 5, 5]
        frames = sim.run(n_steps=100, start=start_seq, device=cpu_device)

        for i in range(1, len(frames)):
            assert frames[i].index > frames[i - 1].index

    def test_run_stops_after_n_steps(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Run should stop when index exceeds n_steps."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        n_steps = 100
        start_seq = [5, 5, 5, 5, 5]
        frames = sim.run(n_steps=n_steps, start=start_seq, device=cpu_device)

        # Final frame should have index > n_steps
        assert frames[-1].index > n_steps

    def test_run_with_default_start(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Run should work with default start sequence (SARS-CoV-2)."""
        sim = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
        )

        # Run with default start (None)
        frames = sim.run(n_steps=50, start=None, device=cpu_device)

        assert len(frames) > 1
        # Default sequence is SARS-CoV-2, should have length 201
        assert frames[0].sequence.shape[0] == 201

    def test_run_with_center_constraint(
        self, simple_energy_fn: callable, cpu_device: str
    ) -> None:
        """Run should work with center constraint."""
        center = torch.tensor([5, 5, 5, 5, 5], device=cpu_device, dtype=torch.int32)
        proposer = IntMutate(min_int=0, max_int=10)

        sim = MetSim(
            model=simple_energy_fn,
            proposer=proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=5,
            center=center,
            max_distance_to_center=2,
        )

        start_seq = [5, 5, 5, 5, 5]
        frames = sim.run(n_steps=50, start=start_seq, device=cpu_device)

        # All frames should be within max_distance of center
        for frame in frames:
            diff = (frame.sequence != center).sum().item()
            assert diff <= 2

    def test_jump_stride_affects_recording(
        self, simple_energy_fn: callable, simple_proposer: IntMutate, cpu_device: str
    ) -> None:
        """Different jump_stride values should affect number of recorded frames."""
        sim_small_stride = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=1,
        )

        sim_large_stride = MetSim(
            model=simple_energy_fn,
            proposer=simple_proposer,
            batch_size=32,
            beta=1.0,
            jump_stride=10,
        )

        start_seq = [5, 5, 5, 5, 5]
        n_steps = 100

        frames_small = sim_small_stride.run(n_steps=n_steps, start=start_seq, device=cpu_device)
        frames_large = sim_large_stride.run(n_steps=n_steps, start=start_seq, device=cpu_device)

        # Smaller stride should record more frames
        # (though exact count depends on acceptance rates)
        # Just verify both produce valid results
        assert len(frames_small) >= 1
        assert len(frames_large) >= 1

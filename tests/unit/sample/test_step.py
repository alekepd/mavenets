"""Tests for mavenets.sample.step module."""

import pytest
import torch

from mavenets.sample.step import (
    State,
    IntMutate,
    BiasedIntMutate,
    _metropolis_crit,
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

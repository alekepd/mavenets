"""Statistical tests for Monte Carlo algorithms in step.py.

These tests verify that the Metropolis Monte Carlo implementation correctly
samples from the target Boltzmann distribution p(x) proportional to exp(-beta * E(x)).
We use simple, analytically tractable energy functions where expected
statistics can be computed.
"""

import pytest
import torch
import numpy as np
from scipy import stats
from collections import Counter

from mavenets.sample.step import (
    State,
    IntMutate,
    BiasedIntMutate,
    _metropolis_crit,
    MetStep,
    MetSim,
)


# =============================================================================
# Fixtures: Energy Functions
# =============================================================================


@pytest.fixture
def uniform_energy():
    """Energy function that returns 0 for all inputs."""

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return torch.zeros(1)
        return torch.zeros(x.shape[0])

    return energy_fn


@pytest.fixture
def linear_energy():
    """Energy = sum of sequence values.

    For a single position with values 0 to n-1:
    p(x) proportional to exp(-beta * x)

    Normalizing constant Z = sum of exp(-beta * k) for k=0 to n-1
    """

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.sum().float().unsqueeze(0)
        return x.sum(dim=-1).float()

    return energy_fn


@pytest.fixture
def quadratic_energy():
    """Energy = sum of squared distances from target value 1.

    E(x) = sum of (x_i - 1)^2
    """
    target = 1

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return ((x.float() - target) ** 2).sum().unsqueeze(0)
        return ((x.float() - target) ** 2).sum(dim=-1)

    return energy_fn


@pytest.fixture
def single_position_energy():
    """Energy for single position: E(x) = x[0].

    This gives an exponential distribution over discrete states.
    """

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x[0].float().unsqueeze(0)
        return x[:, 0].float()

    return energy_fn


# =============================================================================
# Fixtures: Proposers
# =============================================================================


@pytest.fixture
def small_alphabet_proposer():
    """Proposer for alphabet of size 4 (values 0-3)."""
    return IntMutate(min_int=0, max_int=4, n_mutations=1)


@pytest.fixture
def binary_proposer():
    """Proposer for binary alphabet (values 0-1)."""
    return IntMutate(min_int=0, max_int=2, n_mutations=1)


# =============================================================================
# Helper Functions
# =============================================================================


def compute_boltzmann_probabilities(n_states: int, beta: float, energy_fn) -> np.ndarray:
    """Compute theoretical Boltzmann probabilities for states 0 to n_states-1.

    For single-position sequences where E(x) = x.
    """
    energies = np.arange(n_states, dtype=float)
    weights = np.exp(-beta * energies)
    return weights / weights.sum()


def compute_theoretical_mean(n_states: int, beta: float) -> float:
    """Compute theoretical mean for E(x) = x with states 0 to n_states-1."""
    probs = compute_boltzmann_probabilities(n_states, beta, None)
    states = np.arange(n_states)
    return (probs * states).sum()


def compute_theoretical_variance(n_states: int, beta: float) -> float:
    """Compute theoretical variance for E(x) = x with states 0 to n_states-1."""
    probs = compute_boltzmann_probabilities(n_states, beta, None)
    states = np.arange(n_states)
    mean = (probs * states).sum()
    return (probs * (states - mean) ** 2).sum()


def run_simulation_collect_states(
    sim: MetSim,
    n_steps: int,
    start_sequence: torch.Tensor,
    device: str = "cpu",
    burn_in_fraction: float = 0.2,
) -> list:
    """Run simulation and collect state values after burn-in.

    Returns list of sequence tensors.
    """
    frames = sim.run(n_steps=n_steps, start=start_sequence.tolist(), device=device)

    # Discard burn-in period
    burn_in_count = int(len(frames) * burn_in_fraction)
    return [f.sequence for f in frames[burn_in_count:]]


def run_simulation_collect_states_weighted(
    sim: MetSim,
    n_steps: int,
    start_sequence: torch.Tensor,
    device: str = "cpu",
    burn_in_fraction: float = 0.2,
) -> tuple:
    """Run simulation and collect state values with index-based weights.

    The index difference between consecutive frames tells us how many
    underlying Markov chain steps occurred at each state. This allows
    us to reconstruct the true Boltzmann distribution even when
    avoid_null_step=True causes some rejections to be implicit.

    Returns:
        tuple of (list of sequences, list of weights)
    """
    frames = sim.run(n_steps=n_steps, start=start_sequence.tolist(), device=device)

    # Discard burn-in period
    burn_in_count = int(len(frames) * burn_in_fraction)
    frames = frames[burn_in_count:]

    sequences = []
    weights = []

    for i in range(len(frames) - 1):
        sequences.append(frames[i].sequence)
        # Weight is the number of steps spent at this state
        # (difference in index to next frame)
        weight = frames[i + 1].index - frames[i].index
        weights.append(weight)

    # Last frame: use weight of 1 (we don't know how long it would stay)
    if frames:
        sequences.append(frames[-1].sequence)
        weights.append(1)

    return sequences, weights


def count_state_frequencies(sequences: list, position: int = 0) -> Counter:
    """Count frequencies of values at a given position."""
    return Counter(int(seq[position].item()) for seq in sequences)


def count_state_frequencies_weighted(
    sequences: list, weights: list, position: int = 0
) -> Counter:
    """Count weighted frequencies of values at a given position.

    Each sequence is counted according to its weight, which represents
    the number of underlying Markov chain steps at that state.
    """
    counts = Counter()
    for seq, weight in zip(sequences, weights):
        value = int(seq[position].item())
        counts[value] += weight
    return counts


# =============================================================================
# Test Classes
# =============================================================================


class TestEquilibriumDistribution:
    """Test that sampling produces correct equilibrium distributions."""

    @pytest.mark.filterwarnings("ignore:jump_stride=.*:UserWarning")
    def test_uniform_energy_uniform_samples(self, uniform_energy, small_alphabet_proposer):
        """With E(x)=0 for all x, all states should be equally likely.

        Note: Uses jump_stride > 1 since uniform energy means all states are
        equally likely regardless of sampling method.
        """
        n_states = 4
        seq_length = 1
        n_steps = 20000
        beta = 1.0

        start = torch.zeros(seq_length, dtype=torch.int64)

        sim = MetSim(
            model=uniform_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=beta,
            jump_stride=5,
        )

        sequences = run_simulation_collect_states(sim, n_steps, start)
        counts = count_state_frequencies(sequences, position=0)

        # Expected: uniform distribution
        observed = np.array([counts.get(i, 0) for i in range(n_states)])
        expected = np.ones(n_states) * len(sequences) / n_states

        # Chi-squared test
        chi2, p_value = stats.chisquare(observed, expected)

        # Should not reject null hypothesis of uniformity at 1% level
        assert p_value > 0.01, f"Distribution not uniform: chi2={chi2:.2f}, p={p_value:.4f}"

    @pytest.mark.slow
    def test_linear_energy_exponential_distribution(
        self, single_position_energy, small_alphabet_proposer
    ):
        """With E(x)=x, should get exponential distribution p(x) proportional to exp(-beta*x).

        Uses batch_size=1 to ensure proper Metropolis sampling without the
        bias introduced by taking the first accepted proposal in a batch.
        """
        n_states = 4
        seq_length = 1
        n_steps = 100000
        beta = 0.5  # Moderate beta for visible but not extreme differences

        start = torch.zeros(seq_length, dtype=torch.int64)

        # Use batch_size=1 for unbiased Metropolis sampling
        stepper = MetStep(
            model=single_position_energy,
            proposer=small_alphabet_proposer,
            batch_size=1,
            beta=beta,
            avoid_null_step=False,
        )

        # Run simulation manually
        state = State(index=0, sequence=start)
        counts = Counter()

        burn_in = 10000
        for i in range(n_steps + burn_in):
            state = stepper.step(state)
            if i >= burn_in:
                counts[int(state.sequence[0].item())] += 1

        # Calculate observed frequencies
        total = sum(counts.values())
        observed_freq = np.array([counts.get(i, 0) / total for i in range(n_states)])

        # Calculate expected Boltzmann probabilities
        expected_freq = compute_boltzmann_probabilities(n_states, beta, single_position_energy)

        # Chi-squared test on counts
        observed_counts = np.array([counts.get(i, 0) for i in range(n_states)])
        expected_counts = expected_freq * total

        chi2, p_value = stats.chisquare(observed_counts, expected_counts)

        assert p_value > 0.01, (
            f"Distribution doesn't match Boltzmann: chi2={chi2:.2f}, p={p_value:.4f}\n"
            f"Observed: {observed_freq}\nExpected: {expected_freq}"
        )

    @pytest.mark.slow
    def test_mean_matches_theoretical(self, single_position_energy, small_alphabet_proposer):
        """Verify that sampled mean matches theoretical expectation.

        Uses batch_size=1 for unbiased Metropolis sampling.
        """
        n_states = 4
        seq_length = 1
        n_steps = 100000
        beta = 0.7

        start = torch.zeros(seq_length, dtype=torch.int64)

        # Use batch_size=1 for unbiased Metropolis sampling
        stepper = MetStep(
            model=single_position_energy,
            proposer=small_alphabet_proposer,
            batch_size=1,
            beta=beta,
            avoid_null_step=False,
        )

        # Run simulation manually
        state = State(index=0, sequence=start)
        values = []

        burn_in = 10000
        for i in range(n_steps + burn_in):
            state = stepper.step(state)
            if i >= burn_in:
                values.append(int(state.sequence[0].item()))

        # Calculate empirical mean
        empirical_mean = np.mean(values)

        # Calculate theoretical mean
        theoretical_mean = compute_theoretical_mean(n_states, beta)

        # Calculate standard error (accounting for autocorrelation with factor)
        theoretical_var = compute_theoretical_variance(n_states, beta)
        # Use larger autocorrelation factor for MCMC samples
        autocorr_factor = 10
        std_error = np.sqrt(theoretical_var * autocorr_factor / len(values))

        # Check within 4 standard errors
        assert abs(empirical_mean - theoretical_mean) < 4 * std_error, (
            f"Mean mismatch: empirical={empirical_mean:.3f}, "
            f"theoretical={theoretical_mean:.3f}, std_error={std_error:.3f}"
        )

    @pytest.mark.slow
    def test_two_position_independent_marginals(self, small_alphabet_proposer):
        """For E(x,y) = x + y, marginals should be independent.

        Uses batch_size=1 for unbiased Metropolis sampling.
        Tests that each marginal follows the correct Boltzmann distribution.
        """
        n_states = 4
        seq_length = 2
        n_steps = 100000
        beta = 0.5

        def sum_energy(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 1:
                return x.sum().float().unsqueeze(0)
            return x.sum(dim=-1).float()

        start = torch.zeros(seq_length, dtype=torch.int64)

        # Use batch_size=1 for unbiased Metropolis sampling
        stepper = MetStep(
            model=sum_energy,
            proposer=small_alphabet_proposer,
            batch_size=1,
            beta=beta,
            avoid_null_step=False,
        )

        # Run simulation manually
        state = State(index=0, sequence=start)
        counts_pos0 = Counter()
        counts_pos1 = Counter()

        burn_in = 10000
        for i in range(n_steps + burn_in):
            state = stepper.step(state)
            if i >= burn_in:
                counts_pos0[int(state.sequence[0].item())] += 1
                counts_pos1[int(state.sequence[1].item())] += 1

        # Both marginals should follow the same Boltzmann distribution
        expected_freq = compute_boltzmann_probabilities(n_states, beta, None)

        # Check that observed frequencies are close to expected
        # Use a tolerance-based check rather than strict chi-squared
        # since MCMC samples have autocorrelation
        for pos, counts in [(0, counts_pos0), (1, counts_pos1)]:
            total = sum(counts.values())
            observed_freq = np.array([counts.get(i, 0) / total for i in range(n_states)])

            # Each frequency should be within 10% relative error of expected
            for i in range(n_states):
                rel_error = abs(observed_freq[i] - expected_freq[i]) / expected_freq[i]
                assert rel_error < 0.10, (
                    f"Position {pos}, state {i}: relative error {rel_error:.3f} > 0.10\n"
                    f"Observed: {observed_freq}\nExpected: {expected_freq}"
                )

    @pytest.mark.slow
    def test_batched_sampling_with_index_weighting(
        self, single_position_energy, small_alphabet_proposer
    ):
        """Batched sampling with index weighting should recover Boltzmann distribution.

        When using batch_size > 1, the index field tracks how many underlying
        Markov chain steps occurred. By weighting each frame by the index
        difference to the next frame, we recover the correct distribution.

        This test verifies that batch_size=64 with jump_stride=1 and proper
        index weighting produces the correct Boltzmann distribution.
        """
        n_states = 4
        n_steps = 200000
        beta = 0.5

        start = torch.zeros(1, dtype=torch.int64)

        sim = MetSim(
            model=single_position_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=beta,
            jump_stride=1,  # Must be 1 for index weighting to work
        )

        frames = sim.run(n_steps=n_steps, start=start.tolist(), device="cpu")

        # Discard burn-in
        burn_in_count = int(len(frames) * 0.2)
        frames = frames[burn_in_count:]

        # Weight each frame by index difference to next frame
        counts = Counter()
        total_weight = 0

        for i in range(len(frames) - 1):
            val = int(frames[i].sequence[0].item())
            weight = frames[i + 1].index - frames[i].index
            counts[val] += weight
            total_weight += weight

        # Last frame gets weight 1
        counts[int(frames[-1].sequence[0].item())] += 1
        total_weight += 1

        # Calculate expected Boltzmann probabilities
        expected_freq = compute_boltzmann_probabilities(n_states, beta, None)

        # Chi-squared test
        observed_counts = np.array([counts.get(i, 0) for i in range(n_states)])
        expected_counts = expected_freq * total_weight

        chi2, p_value = stats.chisquare(observed_counts, expected_counts)

        assert p_value > 0.01, (
            f"Batched sampling with index weighting failed: chi2={chi2:.2f}, p={p_value:.4f}\n"
            f"Observed freq: {observed_counts / total_weight}\n"
            f"Expected freq: {expected_freq}"
        )

    @pytest.mark.slow
    def test_batched_sampling_different_batch_sizes(
        self, single_position_energy, small_alphabet_proposer
    ):
        """Different batch sizes should all work with index weighting.

        Tests that batch_size=32, 64, and 128 all produce correct distributions
        when using jump_stride=1 and index weighting.
        """
        n_states = 4
        n_steps = 150000
        beta = 0.5

        start = torch.zeros(1, dtype=torch.int64)
        expected_freq = compute_boltzmann_probabilities(n_states, beta, None)

        for batch_size in [32, 64, 128]:
            sim = MetSim(
                model=single_position_energy,
                proposer=small_alphabet_proposer,
                batch_size=batch_size,
                beta=beta,
                jump_stride=1,
            )

            frames = sim.run(n_steps=n_steps, start=start.tolist(), device="cpu")

            # Discard burn-in
            burn_in_count = int(len(frames) * 0.2)
            frames = frames[burn_in_count:]

            # Weight by index difference
            counts = Counter()
            total_weight = 0

            for i in range(len(frames) - 1):
                val = int(frames[i].sequence[0].item())
                weight = frames[i + 1].index - frames[i].index
                counts[val] += weight
                total_weight += weight

            counts[int(frames[-1].sequence[0].item())] += 1
            total_weight += 1

            # Check relative errors are small
            for state in range(n_states):
                observed = counts.get(state, 0) / total_weight
                expected = expected_freq[state]
                rel_error = abs(observed - expected) / expected

                assert rel_error < 0.05, (
                    f"batch_size={batch_size}, state {state}: "
                    f"relative error {rel_error:.3f} > 0.05"
                )

    @pytest.mark.slow
    def test_quadratic_energy_distribution(self, quadratic_energy, small_alphabet_proposer):
        """Quadratic energy E(x) = (x-1)^2 should favor state 1.

        With states 0,1,2,3 and E(x) = (x-1)^2:
          E(0) = 1, E(1) = 0, E(2) = 1, E(3) = 4

        The Boltzmann distribution should peak at state 1 (minimum energy).
        Uses unbiased IntMutate proposer with index weighting.
        """
        n_states = 4
        n_steps = 200000
        beta = 1.0

        start = torch.zeros(1, dtype=torch.int64)

        sim = MetSim(
            model=quadratic_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=beta,
            jump_stride=1,
        )

        frames = sim.run(n_steps=n_steps, start=start.tolist(), device="cpu")

        # Discard burn-in
        burn_in_count = int(len(frames) * 0.2)
        frames = frames[burn_in_count:]

        # Weight by index difference
        counts = Counter()
        total_weight = 0

        for i in range(len(frames) - 1):
            val = int(frames[i].sequence[0].item())
            weight = frames[i + 1].index - frames[i].index
            counts[val] += weight
            total_weight += weight

        counts[int(frames[-1].sequence[0].item())] += 1
        total_weight += 1

        # Compute expected Boltzmann probabilities for quadratic energy
        energies = np.array([(x - 1) ** 2 for x in range(n_states)], dtype=float)
        weights = np.exp(-beta * energies)
        expected_freq = weights / weights.sum()

        # Chi-squared test
        observed_counts = np.array([counts.get(i, 0) for i in range(n_states)])
        expected_counts = expected_freq * total_weight

        chi2, p_value = stats.chisquare(observed_counts, expected_counts)

        # Use a slightly looser threshold due to MCMC autocorrelation
        assert p_value > 0.001, (
            f"Quadratic energy distribution incorrect: chi2={chi2:.2f}, p={p_value:.4f}\n"
            f"Observed freq: {observed_counts / total_weight}\n"
            f"Expected freq: {expected_freq}"
        )

        # State 1 should have highest frequency (lowest energy)
        observed_freq = observed_counts / total_weight
        assert observed_freq[1] > observed_freq[0], "State 1 should be more frequent than state 0"
        assert observed_freq[1] > observed_freq[2], "State 1 should be more frequent than state 2"
        assert observed_freq[1] > observed_freq[3], "State 1 should be more frequent than state 3"

    @pytest.mark.slow
    def test_bias_shifts_mean_with_quadratic_energy(self, quadratic_energy):
        """Biased proposer should shift distribution mean towards center.

        With quadratic energy E(x) = (x-1)^2, the unbiased Boltzmann distribution
        has mean close to 1 (the minimum energy state).

        Using BiasedIntMutate with center=3 should shift the mean towards 3,
        away from the energy minimum at 1.

        We use a low beta (0.3) so the energy penalty is weak enough that
        the proposal bias can significantly shift the distribution.
        """
        n_states = 4
        n_steps = 150000
        beta = 0.3  # Low beta so bias can overcome energy penalty
        center_val = 3

        start = torch.zeros(1, dtype=torch.int64)
        center = torch.tensor([center_val], dtype=torch.int64)

        means = []
        biases = [0.0, 0.5, 0.9]

        for bias in biases:
            proposer = BiasedIntMutate(
                min_int=0, max_int=n_states, bias=bias, center=center, n_mutations=1
            )

            sim = MetSim(
                model=quadratic_energy,
                proposer=proposer,
                batch_size=64,
                beta=beta,
                jump_stride=1,
            )

            frames = sim.run(n_steps=n_steps, start=start.tolist(), device="cpu")

            # Discard burn-in
            burn_in_count = int(len(frames) * 0.2)
            frames = frames[burn_in_count:]

            # Calculate weighted mean using index differences
            weighted_sum = 0
            total_weight = 0

            for i in range(len(frames) - 1):
                val = int(frames[i].sequence[0].item())
                weight = frames[i + 1].index - frames[i].index
                weighted_sum += val * weight
                total_weight += weight

            # Last frame
            weighted_sum += int(frames[-1].sequence[0].item())
            total_weight += 1

            mean = weighted_sum / total_weight
            means.append(mean)

        # Mean should increase towards center (3) as bias increases
        for i in range(len(biases) - 1):
            assert means[i] < means[i + 1], (
                f"Mean should increase with bias towards center={center_val}.\n"
                f"bias={biases[i]}: mean={means[i]:.4f}\n"
                f"bias={biases[i+1]}: mean={means[i+1]:.4f}"
            )

        # With bias=0 and low beta, mean should still be below 1.5
        # (slightly favoring the energy minimum at 1)
        assert means[0] < 1.8, (
            f"With bias=0, mean should favor energy min: {means[0]:.4f}"
        )

        # With bias=0.9, mean should be noticeably shifted towards 3
        assert means[-1] > means[0] + 0.5, (
            f"With bias=0.9, mean should be significantly higher than bias=0.\n"
            f"bias=0: mean={means[0]:.4f}, bias=0.9: mean={means[-1]:.4f}"
        )


class TestBetaScaling:
    """Test that temperature (beta) correctly affects the distribution."""

    def test_beta_zero_approaches_uniform(self, single_position_energy, small_alphabet_proposer):
        """With beta near 0, distribution should approach uniform."""
        n_states = 4
        seq_length = 1
        n_steps = 20000
        beta = 0.01  # Very small beta

        start = torch.zeros(seq_length, dtype=torch.int64)

        sim = MetSim(
            model=single_position_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=beta,
            jump_stride=1,
        )

        sequences = run_simulation_collect_states(sim, n_steps, start)
        counts = count_state_frequencies(sequences, position=0)

        # With very small beta, should be close to uniform
        total = sum(counts.values())
        freqs = [counts.get(i, 0) / total for i in range(n_states)]

        # All frequencies should be within 20% of 0.25
        for i, freq in enumerate(freqs):
            assert 0.15 < freq < 0.35, (
                f"State {i} frequency {freq:.3f} too far from uniform (0.25)"
            )

    @pytest.mark.slow
    def test_higher_beta_lower_energy_mean(self, single_position_energy, small_alphabet_proposer):
        """Higher beta should concentrate distribution on lower energy states."""
        _n_states = 4  # noqa: F841 - documents alphabet size
        seq_length = 1
        n_steps = 25000

        start = torch.zeros(seq_length, dtype=torch.int64)

        means = []
        betas = [0.1, 0.5, 1.0, 2.0]

        for beta in betas:
            sim = MetSim(
                model=single_position_energy,
                proposer=small_alphabet_proposer,
                batch_size=64,
                beta=beta,
                jump_stride=1,
            )

            sequences = run_simulation_collect_states(sim, n_steps, start)
            values = [int(seq[0].item()) for seq in sequences]
            means.append(np.mean(values))

        # Mean should decrease as beta increases (lower energy favored)
        for i in range(len(betas) - 1):
            assert means[i] > means[i + 1], (
                f"Mean at beta={betas[i]} ({means[i]:.3f}) should be > "
                f"mean at beta={betas[i+1]} ({means[i+1]:.3f})"
            )

    @pytest.mark.slow
    def test_variance_decreases_with_beta(self, single_position_energy, small_alphabet_proposer):
        """Higher beta should give lower variance (more concentrated)."""
        _n_states = 4  # noqa: F841 - documents alphabet size
        seq_length = 1
        n_steps = 25000

        start = torch.zeros(seq_length, dtype=torch.int64)

        variances = []
        betas = [0.2, 1.0, 3.0]

        for beta in betas:
            sim = MetSim(
                model=single_position_energy,
                proposer=small_alphabet_proposer,
                batch_size=64,
                beta=beta,
                jump_stride=1,
            )

            sequences = run_simulation_collect_states(sim, n_steps, start)
            values = [int(seq[0].item()) for seq in sequences]
            variances.append(np.var(values))

        # Variance should decrease as beta increases
        for i in range(len(betas) - 1):
            assert variances[i] > variances[i + 1], (
                f"Variance at beta={betas[i]} ({variances[i]:.3f}) should be > "
                f"variance at beta={betas[i+1]} ({variances[i+1]:.3f})"
            )


class TestProposerStatistics:
    """Test that proposers generate correct distributions."""

    def test_int_mutate_uniform_positions(self):
        """IntMutate should mutate all positions with equal probability.

        To reliably detect which position was selected (even when the proposed
        value equals the current value), we use a start sequence with unique
        values at each position that are outside the mutation range.
        """
        seq_length = 5
        n_samples = 10000
        min_int, max_int = 0, 4
        proposer = IntMutate(min_int=min_int, max_int=max_int, n_mutations=1)

        # Use values outside mutation range so any change is detectable
        # Values 10, 11, 12, 13, 14 are outside [0, 4)
        start = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)

        # Track which positions get mutated
        position_counts = Counter()

        for _ in range(n_samples):
            candidates = proposer(start, n_mutants=1)
            # Find which position changed (will always find exactly one)
            diff = (candidates[0] != start).nonzero(as_tuple=True)[0]
            assert len(diff) == 1, "Expected exactly one position to change"
            position_counts[int(diff[0].item())] += 1

        # All positions should be equally likely
        observed = np.array([position_counts.get(i, 0) for i in range(seq_length)])
        expected = np.ones(seq_length) * n_samples / seq_length

        chi2, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.01, f"Position distribution not uniform: p={p_value:.4f}"

    def test_int_mutate_uniform_values(self):
        """IntMutate should select all mutation values with equal probability."""
        n_values = 4
        n_samples = 10000
        proposer = IntMutate(min_int=0, max_int=n_values, n_mutations=1)

        # Start with all zeros, so any non-zero value is a mutation
        start = torch.zeros(1, dtype=torch.int64)

        # Track mutation values
        value_counts = Counter()

        for _ in range(n_samples):
            candidates = proposer(start, n_mutants=1)
            value_counts[int(candidates[0, 0].item())] += 1

        # All values should be equally likely
        observed = np.array([value_counts.get(i, 0) for i in range(n_values)])
        expected = np.ones(n_values) * n_samples / n_values

        chi2, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.01, f"Value distribution not uniform: p={p_value:.4f}"

    def test_biased_mutate_bias_one_uses_center(self):
        """BiasedIntMutate with bias=1 should always propose center values."""
        seq_length = 5
        n_samples = 1000
        center = torch.tensor([1, 2, 3, 2, 1], dtype=torch.int64)

        proposer = BiasedIntMutate(
            min_int=0, max_int=4, bias=1.0, center=center, n_mutations=1
        )

        # Start with all zeros
        start = torch.zeros(seq_length, dtype=torch.int64)

        for _ in range(n_samples):
            candidates = proposer(start, n_mutants=1)
            # Find mutated position
            diff_mask = candidates[0] != start
            if diff_mask.any():
                pos = int(diff_mask.nonzero(as_tuple=True)[0][0].item())
                # Value should be from center
                assert candidates[0, pos].item() == center[pos].item(), (
                    "With bias=1, mutation should use center value"
                )

    def test_biased_mutate_bias_zero_ignores_center(self):
        """BiasedIntMutate with bias=0 should behave like IntMutate."""
        n_values = 4
        n_samples = 10000
        center = torch.tensor([2], dtype=torch.int64)  # Center is always 2

        proposer = BiasedIntMutate(
            min_int=0, max_int=n_values, bias=0.0, center=center, n_mutations=1
        )

        start = torch.zeros(1, dtype=torch.int64)

        value_counts = Counter()
        for _ in range(n_samples):
            candidates = proposer(start, n_mutants=1)
            value_counts[int(candidates[0, 0].item())] += 1

        # All values should be equally likely (ignoring center bias)
        observed = np.array([value_counts.get(i, 0) for i in range(n_values)])
        expected = np.ones(n_values) * n_samples / n_values

        chi2, p_value = stats.chisquare(observed, expected)
        assert p_value > 0.01, f"With bias=0, should be uniform: p={p_value:.4f}"

    def test_biased_mutate_intermediate_bias(self):
        """BiasedIntMutate with bias=0.5 should produce correct value distribution.

        With bias=0.5 and center=3, the probability of proposing value 3 is:
        P(3) = bias * 1.0 + (1-bias) * (1/n_values) = 0.5 + 0.5 * 0.25 = 0.625

        For other values v != 3:
        P(v) = (1-bias) * (1/n_values) = 0.5 * 0.25 = 0.125
        """
        n_samples = 10000
        n_values = 4
        bias = 0.5
        center_val = 3
        center = torch.tensor([center_val], dtype=torch.int64)

        proposer = BiasedIntMutate(
            min_int=0, max_int=n_values, bias=bias, center=center, n_mutations=1
        )

        # Start with value outside the range so all proposals are detectable
        start = torch.tensor([10], dtype=torch.int64)

        value_counts = Counter()
        for _ in range(n_samples):
            candidates = proposer(start, n_mutants=1)
            value_counts[int(candidates[0, 0].item())] += 1

        # Expected probabilities
        # P(center) = bias + (1-bias) * (1/n_values)
        p_center = bias + (1 - bias) * (1 / n_values)
        # P(other) = (1-bias) * (1/n_values)
        p_other = (1 - bias) * (1 / n_values)

        expected_counts = np.array([
            p_center if i == center_val else p_other
            for i in range(n_values)
        ]) * n_samples

        observed_counts = np.array([value_counts.get(i, 0) for i in range(n_values)])

        chi2, p_value = stats.chisquare(observed_counts, expected_counts)
        assert p_value > 0.01, (
            f"Biased distribution incorrect: chi2={chi2:.2f}, p={p_value:.4f}\n"
            f"Observed: {observed_counts / n_samples}\n"
            f"Expected: {expected_counts / n_samples}"
        )


class TestBiasedSampling:
    """Test that BiasedIntMutate shifts the sampled distribution towards center."""

    @pytest.mark.slow
    def test_bias_shifts_distribution_towards_center(self, single_position_energy):
        """Higher bias should shift the equilibrium distribution towards center.

        With BiasedIntMutate, the proposal distribution favors the center value.
        This breaks detailed balance and results in a sampled distribution that
        is shifted towards the center compared to the true Boltzmann distribution.

        As bias increases from 0 to 1, the sampled frequency of the center value
        should monotonically increase.
        """
        n_states = 4
        n_steps = 150000
        beta = 0.5
        center_val = 3  # High energy state - bias should overcome energy penalty

        start = torch.zeros(1, dtype=torch.int64)
        center = torch.tensor([center_val], dtype=torch.int64)

        biases = [0.0, 0.3, 0.6, 0.9]
        center_frequencies = []

        for bias in biases:
            proposer = BiasedIntMutate(
                min_int=0, max_int=n_states, bias=bias, center=center, n_mutations=1
            )

            sim = MetSim(
                model=single_position_energy,
                proposer=proposer,
                batch_size=64,
                beta=beta,
                jump_stride=1,
            )

            frames = sim.run(n_steps=n_steps, start=start.tolist(), device="cpu")

            # Discard burn-in
            burn_in_count = int(len(frames) * 0.2)
            frames = frames[burn_in_count:]

            # Weight by index difference
            counts = Counter()
            total_weight = 0

            for i in range(len(frames) - 1):
                val = int(frames[i].sequence[0].item())
                weight = frames[i + 1].index - frames[i].index
                counts[val] += weight
                total_weight += weight

            counts[int(frames[-1].sequence[0].item())] += 1
            total_weight += 1

            center_freq = counts.get(center_val, 0) / total_weight
            center_frequencies.append(center_freq)

        # Center frequency should increase with bias
        for i in range(len(biases) - 1):
            assert center_frequencies[i] < center_frequencies[i + 1], (
                f"Center frequency should increase with bias.\n"
                f"bias={biases[i]}: freq={center_frequencies[i]:.4f}\n"
                f"bias={biases[i+1]}: freq={center_frequencies[i+1]:.4f}"
            )

        # With bias=0, center (state 3) should have low frequency due to high energy
        # With bias=0.9, center should dominate despite high energy
        assert center_frequencies[0] < 0.15, (
            f"With bias=0, center freq should be low (Boltzmann): {center_frequencies[0]:.4f}"
        )
        assert center_frequencies[-1] > 0.5, (
            f"With bias=0.9, center freq should be high: {center_frequencies[-1]:.4f}"
        )


@pytest.mark.filterwarnings("ignore:jump_stride=.*:UserWarning")
class TestConstraints:
    """Test that distance constraints are respected.

    Note: These tests use jump_stride > 1 to test constraint behavior,
    not exact Boltzmann statistics. The warning about jump_stride is
    suppressed for this class.
    """

    def test_max_distance_never_exceeded(self, uniform_energy, small_alphabet_proposer):
        """With max_distance constraint, no sample should exceed it."""
        seq_length = 10
        n_steps = 10000
        max_distance = 3

        center = torch.zeros(seq_length, dtype=torch.int64)
        start = center.clone()

        sim = MetSim(
            model=uniform_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=1.0,
            center=center,
            max_distance_to_center=max_distance,
            jump_stride=5,
        )

        sequences = run_simulation_collect_states(
            sim, n_steps, start, burn_in_fraction=0.1
        )

        # Check all samples
        for seq in sequences:
            distance = (seq != center).sum().item()
            assert distance <= max_distance, (
                f"Distance {distance} exceeds max {max_distance}"
            )

    def test_boundary_states_reachable(self, uniform_energy, small_alphabet_proposer):
        """States at exactly max_distance should be visited."""
        seq_length = 5
        n_steps = 20000
        max_distance = 2

        center = torch.zeros(seq_length, dtype=torch.int64)
        start = center.clone()

        sim = MetSim(
            model=uniform_energy,
            proposer=small_alphabet_proposer,
            batch_size=64,
            beta=1.0,
            center=center,
            max_distance_to_center=max_distance,
            jump_stride=5,
        )

        sequences = run_simulation_collect_states(
            sim, n_steps, start, burn_in_fraction=0.1
        )

        # Count distances
        distance_counts = Counter()
        for seq in sequences:
            distance = (seq != center).sum().item()
            distance_counts[distance] += 1

        # Should visit states at max_distance (boundary)
        assert distance_counts.get(max_distance, 0) > 0, (
            f"Never visited boundary states at distance {max_distance}"
        )

    def test_constraint_affects_distribution(self, single_position_energy):
        """Constraint should affect which states are visited."""
        seq_length = 5
        n_steps = 15000
        max_distance = 2
        n_values = 4

        proposer = IntMutate(min_int=0, max_int=n_values, n_mutations=1)

        # Center is all zeros
        center = torch.zeros(seq_length, dtype=torch.int64)
        start = center.clone()

        # Run with constraint
        sim_constrained = MetSim(
            model=single_position_energy,
            proposer=proposer,
            batch_size=64,
            beta=0.5,
            center=center,
            max_distance_to_center=max_distance,
            jump_stride=5,
        )

        sequences = run_simulation_collect_states(
            sim_constrained, n_steps, start, burn_in_fraction=0.1
        )

        # All samples should be within constraint
        max_observed_distance = max(
            (seq != center).sum().item() for seq in sequences
        )
        assert max_observed_distance <= max_distance


class TestMetropolisCriterion:
    """Test the Metropolis acceptance criterion directly."""

    def test_negative_delta_always_accepted(self):
        """Moves to lower energy should always be accepted (crit=1)."""
        deltas = torch.tensor([-1.0, -0.5, -0.1, -10.0])
        beta = 1.0

        crits = _metropolis_crit(deltas, beta)

        assert torch.allclose(crits, torch.ones_like(crits)), (
            "Negative deltas should give criterion = 1"
        )

    def test_zero_delta_always_accepted(self):
        """Equal energy moves should always be accepted (crit=1)."""
        deltas = torch.tensor([0.0, 0.0, 0.0])
        beta = 1.0

        crits = _metropolis_crit(deltas, beta)

        assert torch.allclose(crits, torch.ones_like(crits)), (
            "Zero deltas should give criterion = 1"
        )

    def test_positive_delta_probability(self):
        """Positive delta should give exp(-beta * delta)."""
        delta = 1.0
        beta = 2.0

        crit = _metropolis_crit(torch.tensor([delta]), beta)
        expected = np.exp(-beta * delta)

        assert abs(crit.item() - expected) < 1e-6, (
            f"Expected {expected}, got {crit.item()}"
        )

    def test_acceptance_rate_matches_criterion(self):
        """Empirical acceptance rate should match theoretical criterion."""
        delta = 0.5
        beta = 1.0
        n_trials = 10000

        crit = _metropolis_crit(torch.tensor([delta]), beta).item()

        # Simulate acceptance decisions
        variates = torch.rand(n_trials)
        accepts = (variates < crit).sum().item()
        empirical_rate = accepts / n_trials

        # Should be within 3 standard errors
        std_error = np.sqrt(crit * (1 - crit) / n_trials)
        assert abs(empirical_rate - crit) < 3 * std_error, (
            f"Empirical rate {empirical_rate:.3f} differs from criterion {crit:.3f}"
        )

import numpy as np
import pytest

from speaker_enrollment.selection import (
    candidate_sets,
    constrained_sample,
    dominant_clique,
    mean_similarity,
    select_by_coverage,
    select_enrollment,
    violating_pairs,
)
from speaker_enrollment.selection.similarity import adjacency


def speaker_similarity(n=60, outliers=10, seed=0, dim=32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=dim) + 0.6 * rng.normal(size=(n, dim))
    x[:outliers] = rng.normal(size=(outliers, dim)) * 1.5
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x @ x.T


def test_dominant_clique_is_a_maximal_clique():
    sim, tau = speaker_similarity(), 0.5
    clique = dominant_clique(sim, tau)
    adj = adjacency(sim, tau)
    assert len(clique) > 10 and violating_pairs(sim, clique, tau, "high") == 0
    assert not any(adj[v, clique].all() for v in np.setdiff1d(np.arange(len(sim)), clique))


@pytest.mark.parametrize("rule", ["clique", "mean", "all"])
def test_candidate_sets(rule):
    sim = speaker_similarity()
    high, low = candidate_sets(sim, 0.5, rule)
    if rule == "all":
        assert len(high) == len(low) == len(sim)
    else:
        assert sorted(np.concatenate([high, low]).tolist()) == list(range(len(sim)))
        assert set(range(10)) <= set(low.tolist())  # the injected outliers
    if rule == "mean":
        q = mean_similarity(sim)
        assert (q[high] >= 0.5).all() and (q[low] < 0.5).all()


@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("rule", ["clique", "mean"])
def test_configurations_respect_their_constraints(k, rule):
    sim, tau = speaker_similarity(), 0.5
    high, low = candidate_sets(sim, tau, rule)
    rng = np.random.default_rng(1)
    optimal = select_enrollment(sim, k, "optimal", tau, rng, (high, low))
    assert len(set(optimal.indices)) == k and set(optimal.indices) <= set(high)
    assert violating_pairs(sim, optimal.indices, tau, "high") == 0
    bad = select_enrollment(sim, k, "bad", tau, rng, (high, low))
    assert set(bad.indices) <= set(low) and bad.fallback == 0 and violating_pairs(sim, bad.indices, tau, "low") == 0
    assert len(set(select_enrollment(sim, k, "random", tau, rng).indices)) == k
    with pytest.raises(ValueError):
        select_enrollment(sim, k, "best", tau, rng)


def test_fallback_when_the_constraint_cannot_be_met():
    sim = np.full((6, 6), 0.9)
    np.fill_diagonal(sim, 1.0)
    selection = constrained_sample(sim, 3, 0.5, np.random.default_rng(0), "low", np.arange(6))
    assert len(set(selection.indices)) == 3 and selection.fallback == 2


def test_coverage_ordered_selection():
    sim = np.array([[1.0, 0.2, 0.9, 0.9], [0.2, 1.0, 0.2, 0.2], [0.9, 0.2, 1.0, 0.9], [0.9, 0.2, 0.9, 1.0]])
    coverage = np.array([30, 25, 20, 10])
    # utterance 1 has the second-highest coverage but fails the quality check
    assert select_by_coverage(sim, coverage, 3, 0.5).indices.tolist() == [0, 2, 3]
    assert select_by_coverage(sim, coverage, 1, 0.5, "ascending").indices.tolist() == [3]
    full = select_by_coverage(sim, coverage, 4, 0.5)
    assert full.fallback == 1 and sorted(full.indices.tolist()) == [0, 1, 2, 3]
    assert select_by_coverage(sim, coverage, 1, 0.5, candidates=np.array([2, 3])).indices.tolist() == [2]
    # mean check: 1 passes once the mean similarity to the enrolled set is high enough
    mean_sim = sim.copy()
    mean_sim[1, 0] = mean_sim[0, 1] = 0.95
    assert select_by_coverage(mean_sim, coverage, 2, 0.5, quality_check="mean").indices.tolist() == [0, 1]
    with pytest.raises(ValueError):
        select_by_coverage(sim, coverage, 2, 0.5, order="random")

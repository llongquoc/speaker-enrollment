"""Experiment runners, tables and analyses on synthetic embeddings."""

import pandas as pd
import pytest

from speaker_enrollment.experiments import (
    design_choices,
    mean_over_sizes,
    ordering_holds,
    ordering_table,
    run_phoneme_selection,
    run_similarity_selection,
    select_config,
    summarize,
    threshold_sweep,
)
from speaker_enrollment.experiments.tables import (
    icpram_table,
    markdown_table,
    pricai_coverage_table,
    pricai_methods_table,
    ranges,
)
from speaker_enrollment.phonetics import load_phoneme_sets, make_g2p, phonemize_transcripts
from tests.conftest import SENTENCES, synthetic_embeddings


@pytest.fixture
def experiment(tmp_path):
    emb = synthetic_embeddings(n_speakers=8, utterances=(20, 30))
    texts = [SENTENCES[i % len(SENTENCES)] * (1 + i % 3) for i in range(len(emb))]
    table = phonemize_transcripts(pd.DataFrame({"utterance": emb.utterances, "text": texts}), make_g2p("rules"))
    table.to_csv(tmp_path / "phonemes.csv", index=False)
    coverage, sets = load_phoneme_sets(tmp_path / "phonemes.csv", emb.utterances)
    return emb, coverage, sets


def test_random_draws_are_shared_and_reproducible(experiment):
    emb, _, sets = experiment
    kwargs = dict(repetitions=2, candidate_rules=("clique", "mean"), scorings=("mean", "centroid"), phoneme_sets=sets)
    runs = run_similarity_selection(emb, 0.5, **kwargs)
    random_runs = runs[runs.case == "random"]
    clique, mean = (random_runs[random_runs.candidate_rule == r].accuracy.to_numpy() for r in ("clique", "mean"))
    assert (clique == mean).all()
    pd.testing.assert_frame_equal(runs, run_similarity_selection(emb, 0.5, **kwargs))


def test_coverage_ordered_selection(experiment):
    emb, coverage, sets = experiment
    summary = summarize(run_phoneme_selection(emb, 0.5, coverage, sets))
    first = summary[summary.k == 1].set_index("case").coverage_mean
    assert first["high_p"] > first["low_p"]  # highest- versus lowest-coverage utterance


def test_tables(experiment):
    emb, coverage, sets = experiment
    runs = [run_similarity_selection(emb, 0.5, repetitions=2), run_phoneme_selection(emb, 0.5, coverage, sets)]
    summary = summarize(pd.concat(runs, ignore_index=True))
    assert "±" in icpram_table(summary, "optimal")
    assert pricai_methods_table(summary).count("\n") == 11  # header, alignment row, k = 1..10
    assert "High-P P" in pricai_coverage_table(summary)
    table = ordering_table(summary, "icpram2025")
    assert "Optimal > Random" in table and "Random > Bad" in table


def test_ranges_orderings_and_markdown():
    assert ranges([1, 2, 3, 5, 7, 8]) == "1–3, 5, 7–8"
    summary = pd.DataFrame(
        {"case": ["a", "b"] * 3, "k": [1, 1, 2, 2, 3, 3], "accuracy_mean": [0.9, 0.8, 0.7, 0.8, 0.9, 0.9]}
    )
    assert ordering_holds(summary, "a", "b") == [1]
    assert markdown_table(["x", "y"], [["a", "1.0"]]) == "| x | y |\n|---|---:|\n| a | 1.0 |"


def test_threshold_sweep_and_design_choices(experiment):
    emb, coverage, sets = experiment
    sweep, points = threshold_sweep(emb, [0.3, 0.6], coverage, sets, sizes=range(1, 4), repetitions=2, validation=emb)
    assert set(sweep.threshold) == {0.3, 0.6}
    assert set(sweep.case) == {"bad", "random", "optimal", "low_p", "high_p"}
    assert points.validation_far.is_monotonic_decreasing and points.validation_frr.is_monotonic_increasing
    icpram, pricai = design_choices(
        emb, 0.5, coverage, sets, ("clique", "all"), ("mean", "max"), ("pairwise",), sizes=range(1, 4), repetitions=2
    )
    assert len(mean_over_sizes(icpram, ["candidate_rule", "scoring"])) == 2 * 2 * 3
    assert set(pricai.case) == {"low_p", "high_p"} and set(pricai.scoring) == {"mean", "max"}
    subset = select_config(icpram, candidate_rule="all", scoring="max", threshold=0.5 + 1e-12)
    assert len(subset) == 3 * 3 and set(subset.scoring) == {"max"}


def test_threshold_orderings_report():
    from speaker_enrollment.experiments.report import _threshold_orderings

    rows = []
    for tau in (0.1, 0.2):
        for k in (1, 2):
            for case, value in (("high_p", 0.9), ("low_p", 0.8)):
                if (tau, k, case) == (0.2, 2, "low_p"):
                    value = 0.95
                rows.append({"threshold": tau, "k": k, "case": case, "f1_mean": value, "accuracy_mean": value})
    text = "\n".join(_threshold_orderings(pd.DataFrame(rows), ["low_p", "high_p"], calibrated=0.1))
    assert "| Ordering | Accuracy | F1 |" in text
    assert "| High-P > Low-P | all τ except 0.20 | all τ except 0.20 |" in text
    assert "fails at k ≥ 2 only and by at most 0.050" in text

from wcmodel.web_export import _scorecard


def test_scorecard_reports_expected_accuracy_and_draw_gap():
    locked = {
        "fixture": {
            "team_a": "A",
            "team_b": "B",
            "p_a": 0.5,
            "p_draw": 0.3,
            "p_b": 0.2,
        }
    }

    scorecard = _scorecard(locked, {"m:fixture": "D"}, "now")
    summary = scorecard["summary"]

    assert summary["n"] == 1
    assert summary["accuracy"] == 0.0
    assert summary["expected_correct"] == 0.5
    assert summary["actual_draws"] == 1
    assert summary["expected_draws"] == 0.3

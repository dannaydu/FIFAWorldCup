from wcmodel.ingest import get_matches


def test_refresh_failure_keeps_existing_results(tmp_path, monkeypatch):
    path = tmp_path / "results.csv"
    path.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2025-01-01,A,B,1,0,Friendly,True\n"
    )
    monkeypatch.setattr(get_matches, "_try_download_results", lambda _: False)

    matches = get_matches.load_matches(path, refresh=True)

    assert len(matches) == 1
    assert matches.loc[0, "team_a"] == "A"

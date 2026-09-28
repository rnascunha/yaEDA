import logging
from yaeda import TabularEDA


def test_default_log_level_is_warning(classification_df, caplog):
    with caplog.at_level(logging.INFO, logger="yaeda"):
        eda = TabularEDA(df=classification_df, target="target")
        assert eda.log_level == logging.WARNING
        # Since default level is WARNING, no INFO logs should be recorded by the handler
        assert eda.logger.level == logging.WARNING


def test_string_log_level_info(classification_df, caplog):
    with caplog.at_level(logging.INFO, logger="yaeda"):
        eda = TabularEDA(
            df=classification_df,
            target="target",
            preset="minimal",
            log_level="INFO",
        )
        assert eda.logger.level == logging.INFO

        # Trigger profiling and exports
        _ = eda.stats
        _ = eda.to_json()

        # Check that informative progress messages were logged
        records = [r.message for r in caplog.records if r.name == "yaeda"]
        assert any("Initialized TabularEDA" in msg for msg in records)
        assert any("Computing primary table profile" in msg for msg in records)
        assert any("Generating structured JSON metadata" in msg for msg in records)


def test_deep_preset_logs_all_milestones(classification_df, caplog, tmp_path):
    with caplog.at_level(logging.INFO, logger="yaeda"):
        eda = TabularEDA(
            df=classification_df,
            target="target",
            preset="deep",
            log_level=logging.INFO,
        )
        out_html = tmp_path / "logged_report.html"
        eda.to_html(output=out_html)

        log_text = caplog.text
        assert "Computing primary table profile" in log_text
        assert "Computing correlations" in log_text
        assert "Computing feature importance" in log_text
        assert "Computing KMeans clustering" in log_text
        assert "Evaluating pairwise arithmetic interactions" in log_text
        assert "Building interactive HTML dashboard" in log_text
        assert "HTML dashboard generated and saved to" in log_text

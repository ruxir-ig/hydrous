from typer.testing import CliRunner

from hydra_mem.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("download-data", "ingest", "answer", "eval", "judge", "report"):
        assert cmd in result.output


def test_report_command(tmp_path):
    log = tmp_path / "hyp.jsonl.log"
    log.write_text(
        '{"question_id": "a", "question_type": "multi-session", "hypothesis": "x",'
        ' "judgment": true, "judge_response": "yes"}\n'
        '{"question_id": "b_abs", "question_type": "multi-session", "hypothesis": "y",'
        ' "judgment": false, "judge_response": "no"}\n'
    )
    result = runner.invoke(app, ["report", "--log", str(log)])
    assert result.exit_code == 0
    assert "OVERALL" in result.output
    assert "50.0%" in result.output
    assert "abstention" in result.output

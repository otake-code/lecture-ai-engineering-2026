from types import SimpleNamespace

from app import ask


def test_ask_reports_response_time_and_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="回答",
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs == {"model": ask.MODEL, "contents": "質問"}
            return response

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"api_key": "test-key"}
            self.models = FakeModels()

    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask.genai, "Client", FakeClient)
    monkeypatch.setattr(ask.time, "perf_counter", lambda: 10.0)

    assert ask.ask("質問") == "回答"

    output = capsys.readouterr().out
    assert "[応答時間] 0.000秒" in output
    assert "[入力トークン数] 12" in output
    assert "[出力トークン数] 7" in output


def test_compare_runs_every_model_and_question(monkeypatch, capsys):
    measurements = {
        ("model-a", "質問1"): ask.Measurement("model-a", "質問1", "回答1", 1.25, 10, 20),
        ("model-a", "質問2"): ask.Measurement("model-a", "質問2", "回答2", 1.50, 11, 21),
        ("model-b", "質問1"): ask.Measurement("model-b", "質問1", "回答3", 2.25, 12, 22),
        ("model-b", "質問2"): ask.Measurement("model-b", "質問2", "回答4", 2.50, 13, 23),
    }
    monkeypatch.setattr(
        ask,
        "measure",
        lambda question, model=ask.MODEL: measurements[(model, question)],
    )

    ask.compare(["model-a", "model-b"], ["質問1", "質問2"])

    output = capsys.readouterr().out
    assert output.count(" | ") >= 5
    assert "model-a | 質問1 | 1.250 | 10 | 20" in output
    assert "model-b | 質問2 | 2.500 | 13 | 23" in output
    assert "[model-a / 質問1]" in output
    assert "回答4" in output

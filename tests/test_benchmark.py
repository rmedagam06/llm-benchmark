import pytest

from src.benchmark.mmlu import _format_question, extract_answer

# ---- prompt formatting ----


def _make_row(q="What is 2+2?", choices=None, answer=0):
    return {"question": q, "choices": choices or ["2", "4", "6", "8"], "answer": answer}


def test_format_question_zero_shot():
    row = _make_row()
    prompt = _format_question(row)
    assert "Question:" in prompt
    assert "A)" in prompt
    assert "Answer:" in prompt
    assert "2+2" in prompt


def test_format_question_few_shot():
    shot = _make_row(q="1+1?", answer=1)
    row = _make_row()
    prompt = _format_question(row, shots=[shot])
    assert "Answer: B" in prompt  # shot answer filled in
    assert prompt.endswith("Answer:")  # main question unanswered


# ---- answer extraction ----


@pytest.mark.parametrize(
    "response,expected",
    [
        ("A", "A"),
        ("The answer is B.", "B"),
        ("C) Something", "C"),
        ("  D  ", "D"),
        ("The correct choice is C\n", "C"),
        ("none of them", None),
    ],
)
def test_extract_answer(response, expected):
    assert extract_answer(response) == expected


# ---- BenchmarkResult serialisation ----


def test_benchmark_result_to_dict():
    from src.benchmark.runner import BenchmarkResult

    r = BenchmarkResult(
        model_name="test",
        adapter_name=None,
        quant_level=None,
        subjects=["math"],
        num_shots=0,
        num_questions=10,
        num_correct=7,
        accuracy=0.7,
        per_subject_accuracy={"math": 0.7},
        avg_ttft_ms=50.0,
        tokens_per_second=30.0,
        peak_vram_mb=256.0,
    )
    d = r.to_dict()
    assert d["accuracy"] == 0.7
    assert d["num_correct"] == 7
    assert "timestamp" in d

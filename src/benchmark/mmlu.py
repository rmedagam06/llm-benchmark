from __future__ import annotations

CHOICE_LABELS = ["A", "B", "C", "D"]


def _format_question(row: dict, shots: list[dict] | None = None) -> str:
    """Format a single MMLU question with optional few-shot examples prepended."""
    lines: list[str] = []

    if shots:
        for shot in shots:
            lines.append(_format_single(shot, include_answer=True))
            lines.append("")

    lines.append(_format_single(row, include_answer=False))
    return "\n".join(lines)


def _format_single(row: dict, include_answer: bool) -> str:
    choices = row["choices"]
    q = f"Question: {row['question']}"
    opts = "\n".join(f"{CHOICE_LABELS[i]}) {choices[i]}" for i in range(len(choices)))
    if include_answer:
        answer_label = CHOICE_LABELS[row["answer"]]
        return f"{q}\n{opts}\nAnswer: {answer_label}"
    return f"{q}\n{opts}\nAnswer:"


def load_mmlu_subset(
    subjects: list[str],
    num_shots: int = 0,
    split: str = "test",
    max_samples: int = 50,
) -> list[dict]:
    """Load MMLU questions across subjects and return formatted prompt dicts."""
    from datasets import load_dataset

    items: list[dict] = []

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
        ds = ds.select(range(min(max_samples, len(ds))))

        shots: list[dict] = []
        if num_shots > 0:
            dev_ds = load_dataset("cais/mmlu", subject, split="dev", trust_remote_code=True)
            shots = [dev_ds[i] for i in range(min(num_shots, len(dev_ds)))]

        for row in ds:
            items.append(
                {
                    "subject": subject,
                    "prompt": _format_question(row, shots if num_shots > 0 else None),
                    "answer_index": row["answer"],
                    "answer_label": CHOICE_LABELS[row["answer"]],
                }
            )

    return items


def extract_answer(response: str) -> str | None:
    """Extract A/B/C/D from a model response."""
    for line in response.strip().splitlines():
        stripped = line.strip()
        if stripped and stripped[0] in CHOICE_LABELS:
            return stripped[0]
    for ch in response:
        if ch in CHOICE_LABELS:
            return ch
    return None

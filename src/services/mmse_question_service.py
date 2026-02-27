from .correctness_hardcoded import check_hardcoded_correctness
from .scoring_engine import calculate_score


def process_mmse_question(
    question_type: str,
    transcript: str,
    ml_prediction: str,
    caregiver_is_correct: bool | None = None
):
    """
    Processes a single MMSE question.

    - transcript: already generated in main.py
    - ml_prediction: 'Control' or 'Dementia'
    - caregiver_is_correct: used ONLY for orientation questions
    """

    # CASE 1 — Caregiver decides correctness (orientation)
    if caregiver_is_correct is not None:
        correctness = {"answer": caregiver_is_correct}

    # CASE 2 — Backend hardcoded correctness
    else:
        correctness = check_hardcoded_correctness(
            question_type, transcript
        )

    results = []
    total_score = 0

    for item, is_correct in correctness.items():
        score = calculate_score(is_correct, ml_prediction)
        total_score += score

        results.append({
            "item": item,
            "is_correct": is_correct,
            "score": score
        })

    return results, total_score

def calculate_score(is_correct: bool, ml_label: str):
    if not is_correct:
        return 0

    if ml_label == "Control":
        return 1

    return 0.5

from utils.text_normalizer import normalize

ANSWER_KEY = {
    # Repeat / Recall words
    "repeat_words": ["ball", "car", "man"],
    "recall_words": ["ball", "car", "man"],

    # Object naming
    "name_objects": {
        "watch": ["watch", "wristwatch"],
        "pen": ["pen"]
    },

    # Sentence repetition
    "repeat_sentence": [
        "no ifs ands or buts",
        "no ifs and buts"
    ],

    # Spell backward
    "spell_world": ["dlrow"],

    # Serial 7s
    "serial_7": ["93", "86", "79", "72", "65"]
}


def check_hardcoded_correctness(question_type: str, transcript: str):
    transcript = normalize(transcript)

    # Repeat / Recall words
    if question_type in ["repeat_words", "recall_words"]:
        return {
            word: word in transcript
            for word in ANSWER_KEY[question_type]
        }

    # Object naming
    if question_type == "name_objects":
        return {
            obj: any(k in transcript for k in keys)
            for obj, keys in ANSWER_KEY["name_objects"].items()
        }

    # Sentence repetition
    if question_type == "repeat_sentence":
        return {
            "sentence": any(
                normalize(ans) in transcript
                for ans in ANSWER_KEY["repeat_sentence"]
            )
        }

    # Spell WORLD backward
    if question_type == "spell_world":
        return {
            "world_backward": "dlrow" in transcript
        }

    # Serial 7s
    if question_type == "serial_7":
        return {
            num: num in transcript
            for num in ANSWER_KEY["serial_7"]
        }

    raise ValueError("Unsupported hardcoded question type")

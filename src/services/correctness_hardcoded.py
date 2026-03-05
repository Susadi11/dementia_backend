from utils.text_normalizer import normalize
from datetime import datetime

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
        "no ifs and buts",
        "no ifs no buts"
    ],

    # Spell backward
    "spell_world": ["dlrow"],

    # Serial 7s
    "serial_7": ["93", "86", "79", "72", "65"]
}

NUMBER_WORDS = {
    "one": 1, "first": 1,
    "two": 2, "second": 2,
    "three": 3, "third": 3,
    "four": 4, "fourth": 4,
    "five": 5, "fifth": 5,
    "six": 6, "sixth": 6,
    "seven": 7, "seventh": 7,
    "eight": 8, "eighth": 8,
    "nine": 9, "ninth": 9,
    "ten": 10, "tenth": 10,
    "eleven": 11, "eleventh": 11,
    "twelve": 12, "twelfth": 12,
    "thirteen": 13, "thirteenth": 13,
    "fourteen": 14, "fourteenth": 14,
    "fifteen": 15, "fifteenth": 15,
    "sixteen": 16, "sixteenth": 16,
    "seventeen": 17, "seventeenth": 17,
    "eighteen": 18, "eighteenth": 18,
    "nineteen": 19, "nineteenth": 19,
    "twenty": 20, "twentieth": 20,
    "twenty one": 21, "twenty first": 21,
    "twenty two": 22, "twenty second": 22,
    "twenty three": 23, "twenty third": 23,
    "twenty four": 24, "twenty fourth": 24,
    "twenty five": 25, "twenty fifth": 25,
    "twenty six": 26, "twenty sixth": 26,
    "twenty seven": 27, "twenty seventh": 27,
    "twenty eight": 28, "twenty eighth": 28,
    "twenty nine": 29, "twenty ninth": 29,
    "thirty": 30, "thirtieth": 30,
    "thirty one": 31, "thirty first": 31
}


def check_orientation(question_type: str, transcript: str):

    now = datetime.now()

    current_year = str(now.year)
    current_month = normalize(now.strftime("%B"))
    current_date = now.day
    current_day = normalize(now.strftime("%A"))

    transcript = normalize(transcript)

    if question_type == "year":
        return {"year": current_year in transcript}

    if question_type == "month":
        return {"month": current_month in transcript}

    if question_type == "date":
        spoken_number = NUMBER_WORDS.get(transcript)

        # If transcript is numeric like "5"
        if spoken_number is None and transcript.isdigit():
            spoken_number = int(transcript)

        return {"date": spoken_number == current_date}

    if question_type == "day":
        return {"day": current_day in transcript}

    return None


def check_hardcoded_correctness(question_type: str, transcript: str):
    transcript = normalize(transcript)

    # Orientation questions
    orientation_result = check_orientation(question_type, transcript)
    if orientation_result:
        return orientation_result

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
        joined = transcript.replace(" ", "")
        return {
            "world_backward": joined == "dlrow"
        }

    # Serial 7s
    if question_type == "serial_7":
        return {
            num: num in transcript
            for num in ANSWER_KEY["serial_7"]
        }

    raise ValueError("Unsupported hardcoded question type")
import re
import html
import unicodedata
import string


# Chuẩn hóa Unicode tiếng Việt
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


# Xóa khoảng trắng thừa
def clean_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Xóa dấu câu nhưng giữ "_"
def remove_punctuation_keep_underscore(text: str) -> str:

    punctuation = string.punctuation.replace("_", "")

    translator = str.maketrans("", "", punctuation)

    return text.translate(translator)


# Rút ký tự lặp
# nguuuu -> ngu
def normalize_repeated_chars(text: str) -> str:

    return re.sub(r"(.)\1{1,}", r"\1", text)


# Rút từ lặp
# ngu ngu ngu -> ngu
def normalize_repeated_words(text: str) -> str:

    return re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text)


# Main preprocessing
def preprocess_text(text: str) -> str:

    text = str(text)

    # decode html
    text = html.unescape(text)

    # unicode normalize
    text = normalize_unicode(text)

    # lowercase
    text = text.lower()

    # normalize repeated chars
    text = normalize_repeated_chars(text)

    # remove punctuation
    text = remove_punctuation_keep_underscore(text)

    # clean spaces
    text = clean_whitespace(text)

    # normalize repeated words
    text = normalize_repeated_words(text)

    return text
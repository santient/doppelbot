import re
import unicodedata


def unicode_to_ascii(s):
    # convert to ascii
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # discard unnecessary characters
    s = unicode_to_ascii(s.strip())
    s = re.sub(r"([.!?])", r"\1", s)
    s = re.sub(r"[^a-zA-Z.!?,'""0-9]+", r" ", s)
    return s


def read_lines(path="dialogues_text.txt"):
    print("Reading Lines...")

    # read the file
    try:
        src = open(path, 'r', encoding='utf-8').read().split('\n')
    except:
        raise FileNotFoundError
    src = ["<SOM>" + normalize_string(s) + "<EOM>" for s in src]
    return src
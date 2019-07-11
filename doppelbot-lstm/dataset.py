import re
import string
import unicodedata
import torch
import tqdm
from torch.utils.data import Dataset


CHARS = ["<PAD>", "<SOM>", "<EOM>", "<SOW>", "<EOW>"] + list(string.printable)

class ConversationDataset(Dataset):
    def __init__(self, path, max_len, device):
        self.path = path
        self.max_len = max_len
        self.device = device
        self.src = read_lines(path)

    def __len__(self):
        return len(self.src) - self.max_len

    def __getitem__(self, idx):
        lines = self.src[idx:idx + self.max_len]
        cnv = []
        for line in lines:
            msg = [[CHARS.index("<SOM>")]]
            for word in line.split(" "):
                wrd = [CHARS.index("<SOW>")]
                for char in word:
                    wrd.append(CHARS.index(char))
                wrd.append(CHARS.index("<EOW>"))
                msg.append(wrd)
            msg.append([CHARS.index("<EOM>")])
            cnv.append([torch.tensor(wrd, dtype=torch.long, device=self.device).unsqueeze(0) for wrd in msg])
        X = cnv[:-1]
        Y = torch.cat(list(flatten(cnv[-1])), dim=-1)[0]
        return X, Y

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

def read_lines(path):
    # read the file
    try:
        src = open(path, 'r', encoding='utf-8').read().split('\n')
    except FileNotFoundError:
        raise
    src = [normalize_string(s) for s in src if len(s) > 0]
    return src

def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

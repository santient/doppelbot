import torch
from torch import nn
import torch.nn.functional as F
import dataset


class Encoder(nn.Module):
    def __init__(self, char_latent_dim, word_latent_dim, msg_latent_dim, name_latent_dim, convo_latent_dim):
        self.char_embedding = nn.Embedding(char_latent_dim)
        self.char_lstm = nn.LSTM(char_latent_dim, word_latent_dim, num_layers=4, dropout=1/4)
        self.word_lstm = nn.LSTM(word_latent_dim, msg_latent_dim, num_layers=4, dropout=1/3)
        self.name_msg_fusion = nn.Bilinear(name_latent_dim, msg_latent_dim, msg_latent_dim)
        self.msg_lstm = nn.LSTM(msg_latent_dim, convo_latent_dim, num_layers=4, dropout=1/2)

    def forward(self, convo):
        char_embed = [(name, [self.char_embedding(word) for word in msg]) for name, msg in convo]
        encoded_words = [(name, torch.stack([self.char_lstm(word)[-1] for word in msg])) for name, msg in char_embed]
        encoded_msg = [(name, self.word_lstm(msg)[-1]) for name, msg in encoded_words]
        encoded_msg_fusion = torch.stack([self.name_msg_fusion(name, msg) for name, msg in encoded_msg])
        encoded_convo = self.msg_lstm(encoded_msg_fusion)[-1]
        return encoded_convo

class Generator(nn.Module):
    def __init__(self, char_latent_dim, word_latent_dim, msg_latent_dim):
        self.word_lstm = nn.LSTM(msg_latent_dim, word_latent_dim, num_layers=4)
        self.char_lstm = nn.LSTM(word_latent_dim, char_latent_dim, num_layers=4)
        self.softmax = nn.Softmax()
        self.word_state = None
        self.char_state = None

    def forward(self, encoded):
        if self.train:
            
        else:
            msg = []
            word_out = encoded
            char_out_argmax = dataset.chars.index("start")
            while dataset.chars[char_out_argmax] != "send":
                

class Chatbot(nn.Module):
    def __init__(self, char_latent_dim, word_latent_dim, msg_latent_dim, name_latent_dim, convo_latent_dim):
        self.encoder = Encoder(char_latent_dim, word_latent_dim, msg_latent_dim, name_latent_dim, convo_latent_dim)
        self.generator = Generator(char_latent_dim, word_latent_dim, msg_latent_dim)

    def forward(self, convo):
        encoded = self.encoder(convo)
        if self.train:
            pass
        else:
            return self.generator(encoded)


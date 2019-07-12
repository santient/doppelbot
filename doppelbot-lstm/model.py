import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_chars, char_latent_dim, word_latent_dim, msg_latent_dim):
        super(Encoder, self).__init__()
        self.char_embedding = nn.Embedding(num_chars, char_latent_dim)
        self.char_lstm = nn.LSTM(char_latent_dim, word_latent_dim, num_layers=4, dropout=1/4)
        self.word_lstm = nn.LSTM(word_latent_dim, msg_latent_dim, num_layers=4, dropout=1/3)
        self.reset()

    def reset(self):
        self.context = None

    def forward(self, msg):
        char_embed = [self.char_embedding(word).transpose(0, 1) for word in msg]
        encoded_words = torch.stack([self.char_lstm(word)[0][-1] for word in char_embed])
        encoded_msg, self.context = self.word_lstm(encoded_words, self.context)
        return encoded_msg

class Generator(nn.Module):
    def __init__(self, num_chars, char_latent_dim, word_latent_dim, msg_latent_dim):
        super(Generator, self).__init__()
        self.msg_to_word = nn.Linear(msg_latent_dim, word_latent_dim)
        self.word_lstm = nn.LSTM(word_latent_dim, word_latent_dim, num_layers=4)
        self.word_to_char = nn.Linear(word_latent_dim, char_latent_dim)
        self.char_lstm = nn.LSTM(char_latent_dim, char_latent_dim, num_layers=4)
        self.char_to_prob = nn.Linear(char_latent_dim, num_chars)
        self.softmax = nn.Softmax(dim=-1)
        self.reset()

    def reset(self):
        self.reset_char()
        self.reset_word()

    def reset_char(self):
        self.char_out = None
        self.char_state = None

    def reset_word(self):
        self.word_out = None
        self.word_state = None

    def forward(self, encoded=None):
        if encoded is not None:
            self.reset_word()
            self.word_out = self.msg_to_word(encoded)
        elif self.word_out is None:
            raise ValueError
        if self.char_out is None:
            self.reset_char()
            self.word_out, self.word_state = self.word_lstm(self.word_out, self.word_state)
            self.char_out = self.word_to_char(self.word_out)
        self.char_out, self.char_state = self.char_lstm(self.char_out, self.char_state)
        return self.softmax(self.char_to_prob(self.char_out[-1]))

class Chatbot(nn.Module):
    def __init__(self, num_chars, char_latent_dim, word_latent_dim, msg_latent_dim):
        super(Chatbot, self).__init__()
        self.encoder = Encoder(num_chars, char_latent_dim, word_latent_dim, msg_latent_dim)
        self.generator = Generator(num_chars, char_latent_dim, word_latent_dim, msg_latent_dim)

    def reset(self):
        self.encoder.reset()
        self.generator.reset()

    def forward(self, convo=None, encoded=None):
        # TODO deprecate
        if convo is not None:
            for msg in convo:
                encoded = self.encoder(msg)
        if encoded is not None:
            return self.generator(encoded)
        else:
            raise ValueError

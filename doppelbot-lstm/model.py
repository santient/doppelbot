import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_chars, char_latent_dim, word_latent_dim, msg_latent_dim):
        super()
        self.char_embedding = nn.Embedding(num_chars, char_latent_dim)
        self.char_lstm = nn.LSTM(char_latent_dim, word_latent_dim, num_layers=4, dropout=1/4)
        self.word_lstm = nn.LSTM(word_latent_dim, msg_latent_dim, num_layers=4, dropout=1/3)
        # self.name_msg_fusion = nn.Bilinear(name_latent_dim, msg_latent_dim, msg_latent_dim)
        # self.msg_lstm = nn.LSTM(msg_latent_dim, ctxt_latent_dim, num_layers=4, dropout=1/2)
        self.reset()

    def reset(self):
        self.context = None

    def forward(self, msg):
        char_embed = [self.char_embedding(word) for word in msg]
        encoded_words = [(name, torch.stack([self.char_lstm(word)[-1] for word in msg])) for name, msg in char_embed]
        encoded_msg, self.context = self.word_lstm(encoded_words, self.context)
        # encoded_msg = [(name, self.word_lstm(msg)[-1]) for name, msg in encoded_words]
        # encoded_msg_fusion = torch.stack([self.name_msg_fusion(name, msg) for name, msg in encoded_msg])
        # encoded = self.msg_lstm(encoded_msg_fusion, self.context)[-1]
        return encoded_msg

class Generator(nn.Module):
    def __init__(self, char_latent_dim, word_latent_dim, msg_latent_dim):
        super()
        self.word_lstm = nn.LSTM(msg_latent_dim, word_latent_dim, num_layers=4)
        self.char_lstm = nn.LSTM(word_latent_dim, char_latent_dim, num_layers=4)
        self.softmax = nn.Softmax()
        self.reset()

    def reset(self):
        self.word_out = None
        self.word_state = None
        self.char_out = None
        self.char_state = None

    def forward(self, encoded):
        if self.char_out is None:
            self.char_state = None
            self.word_out, self.word_state = self.word_lstm(encoded, self.word_state)
            self.char_out = self.word_out
            self.char_state = None
        self.char_out = self.char_lstm(self.char_out, self.char_state)
        return self.char_out
        # TODO use below code in training and generation
        # if self.train:
        #     pass  # TODO
        # else:
        #     assert encoded.dims == 1
        #     encoded = encoded.unsqueeze(0)
        #     generated_msg = []
        #     word_out = encoded
        #     word_state = None
        #     next_char = "start"
        #     while next_char != "send":
        #         word_out, word_state = self.word_lstm(word_out, word_state)
        #         char_out = word_out
        #         char_state = None
        #         while next_char != "send" and next_char != " ":
        #             char_out, char_state = self.char_lstm(char_out, char_state)
        #             char_out_sm = self.softmax(char_out)
        #             generated_msg.append(char_out_sm)
        #             next_char = dataset.CHARS[char_out_sm.argmax().value()]

class Chatbot(nn.Module):
    def __init__(self, num_chars, char_latent_dim, word_latent_dim, msg_latent_dim):
        super()
        self.encoder = Encoder(num_chars, char_latent_dim, word_latent_dim, msg_latent_dim)
        self.generator = Generator(char_latent_dim, word_latent_dim, msg_latent_dim)

    def forward(self, convo=None, encoded=None):
        if convo is not None:
            encoded = self.encoder(convo)
        if encoded is not None:
            return self.generator(encoded)
        else:
            raise ValueError

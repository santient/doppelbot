import torch
from torch.backends import cudnn
import dataset, model


def to_chars(idxs):
    idxs = list(dataset.flatten(idxs))
    chars = [dataset.CHARS[idx] for idx in idxs]
    return chars

def generate():
    cudnn.benchmark = True
    device = torch.device("cuda:0")
    chatbot = model.Chatbot(len(dataset.CHARS), 16, 64, 256).to(device)
    chatbot.load_state_dict(torch.load("/home/santiago/Projects/DoppelBot/lstm/checkpoints/model_001.pth"))

    # generation loop
    while True:
        query = input("Input: ")
        query = dataset.normalize_string(query)
        msg = [[dataset.CHARS.index("<SOM>")]]
        for word in query.split(" "):
            wrd = [dataset.CHARS.index("<SOW>")]
            for char in word:
                wrd.append(dataset.CHARS.index(char))
            wrd.append(dataset.CHARS.index("<EOW>"))
            msg.append(wrd)
        msg.append([dataset.CHARS.index("<EOM>")])
        print("You:", to_chars(msg))
        x = [torch.tensor(wrd, dtype=torch.long, device=device).unsqueeze(0) for wrd in msg]
        encoded = chatbot.encoder(x)
        chatbot.encoder.reset()
        generated = []
        max_len = 128
        while len(generated) < max_len:
            char = torch.argmax(chatbot.generator(encoded)).item()
            generated.append(char)
            if dataset.CHARS[char] == "<EOW>":
                chatbot.generator.reset_char()
            elif dataset.CHARS[char] == "<EOM>":
                break
        chatbot.generator.reset()
        print("DoppelBot:", to_chars(generated))

if __name__ == "__main__":
    generate()

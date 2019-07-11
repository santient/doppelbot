import random
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import dataset, model


def train():
    cudnn.benchmark = True
    device = torch.device("cuda:0")
    chatbot = model.Chatbot(len(dataset.CHARS), 16, 64, 256).to(device)
    opt = torch.optim.Adam(chatbot.parameters(), lr=1e-3)
    dset = dataset.ConversationDataset("/home/santiago/Data/daily_dialog/train/dialogues_train.txt", 2, device)

    # training loop
    chatbot.train()
    epochs = 100
    idxs = list(range(len(dset)))
    for epoch in range(epochs):
        random.shuffle(idxs)
        for step, idx in enumerate(idxs):
            X, Y = dset[idx]
            losses = []
            for i in range(1, len(Y)):
                msg = []
                for char in Y[:i]:
                    if dataset.CHARS[char] in ["<SOM>", "<EOM>", "<SOW>"]:
                        msg.append([char])
                    else:
                        msg[-1].append(char)
                msg = [torch.tensor(wrd, dtype=torch.long, device=device).unsqueeze(0) for wrd in msg]
                convo = X + [msg]
                target = torch.tensor([Y[i]], dtype=torch.long, device=device)
                generated = chatbot(convo=convo)
                loss = F.cross_entropy(generated, target)
                losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                chatbot.reset()
            print("(Epoch {}/{}) (Step {}/{}) (Average Loss {})".format(
                epoch, epochs, step, len(dset), sum(losses) / len(losses)
            ))
        torch.save(chatbot.state_dict(), "/home/santiago/Projects/DoppelBot/lstm/checkpoints/model_{}.pth".format(epoch))
        print("Epoch {} complete!".format(epoch))
    print("Training complete!")


if __name__ == "__main__":
    train()

import random
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import dataset, model


def train():
    cudnn.benchmark = True
    device = torch.device("cuda:0")
    chatbot = model.Chatbot(len(dataset.CHARS), 64, 64, 256).to(device)
    opt = torch.optim.Adam(chatbot.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dset = dataset.ConversationDataset("/home/santiago/Data/daily_dialog/train/dialogues_train.txt", 2, device)

    # training loop
    chatbot.train()
    epochs = 100
    idxs = list(range(len(dset)))
    for epoch in range(epochs):
        random.shuffle(idxs)
        for step, idx in enumerate(idxs):
            X, Y = dset[idx]
            assert len(X) > 0
            assert Y.shape[0] == 1
            assert Y.shape[1] > 0
            for msg in X:
                encoded = chatbot.encoder(msg)
            chatbot.encoder.reset()
            generated = []
            for char in Y[0].tolist():
                generated.append(chatbot.generator(encoded))
                if dataset.CHARS[char] == "<EOW>":
                    chatbot.generator.reset_char()
                elif dataset.CHARS[char] == "<EOM>":
                    chatbot.generator.reset()
            generated = torch.stack(generated, dim=1)
            loss = loss_fn(generated[0], Y[0])
            opt.zero_grad()
            loss.backward()
            opt.step()
            # losses = []
            # for i in range(1, len(Y)):
            #     msg = []
            #     for char in Y[:i]:
            #         if dataset.CHARS[char] in ["<SOM>", "<EOM>", "<SOW>"]:
            #             msg.append([char])
            #         else:
            #             msg[-1].append(char)
            #     msg = [torch.tensor(wrd, dtype=torch.long, device=device).unsqueeze(0) for wrd in msg]
            #     convo = X + [msg]
            #     target = torch.tensor([Y[i]], dtype=torch.long, device=device)
            #     generated = chatbot(convo=convo)
            #     loss = F.cross_entropy(generated, target)
            #     losses.append(loss.item())
            #     opt.zero_grad()
            #     loss.backward()
            #     opt.step()
            #     chatbot.reset()
            print("(Epoch {}/{}) (Step {}/{}) (Loss {})".format(
                epoch, epochs, step, len(dset), loss.item()
            ))
        torch.save(chatbot.state_dict(), "/home/santiago/Projects/DoppelBot/lstm/checkpoints/model_{}.pth".format(epoch))
        torch.save(opt.state_dict(), "/home/santiago/Projects/DoppelBot/lstm/checkpoints/opt_{}.pth".format(epoch))
        print("Epoch {} complete!".format(epoch))
    print("Training complete!")

if __name__ == "__main__":
    train()

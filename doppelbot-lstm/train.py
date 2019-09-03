import random
import torch
import torch.nn.functional as F
from torch.backends import cudnn
import dataset, model


def train():
    cudnn.benchmark = True
    device = torch.device("cuda:0")
    chatbot = model.Chatbot(len(dataset.CHARS), 64, 256, 1024).to(device)
    opt = torch.optim.Adam(chatbot.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    dset = dataset.ConversationDataset("/home/santiago/Data/daily_dialog/train/dialogues_train.txt", 2, device)

    # training loop
    chatbot.train()
    epochs = 100
    idxs = list(range(len(dset)))
    for epoch in range(epochs):
        epoch += 1
        random.shuffle(idxs)
        for step, idx in enumerate(idxs):
            step += 1
            X, Y = dset[idx]
            assert len(X) > 0
            assert Y.shape[0] == 1
            assert Y.shape[1] > 0
            for msg in X:
                encoded = chatbot.encoder(msg)
            context = chatbot.encoder.context
            avg_loss = 0
            for i in range(1, Y.shape[1] + 1):
                chatbot.encoder.context = context
                force = dataset.hierarchy(Y[:, :i])[0]
                generated = chatbot.generator(encoded)
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
                chatbot.reset()
            print("(Epoch {}/{}) (Step {}/{}) (Average Loss {})".format(
                epoch, epochs, step, len(dset), avg_loss.item()
            ))
        torch.save(chatbot.state_dict(), "/home/santiago/Projects/DoppelBot/lstm/checkpoints/model_{:03d}.pth".format(epoch))
        torch.save(opt.state_dict(), "/home/santiago/Projects/DoppelBot/lstm/checkpoints/opt_{:03d}.pth".format(epoch))
        print("Epoch {} complete!".format(epoch))
    print("Training complete!")

if __name__ == "__main__":
    train()

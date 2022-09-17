import time
import sys
import torch
import random
import argparse

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='GRU')
    return parser

class LSTMModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 128)
        self.rnn = torch.nn.LSTM(128, 256, batch_first=True)
        self.linear = torch.nn.Linear(256, 10)

    def forward(self, sentence, state=None):
        embed = self.embed(sentence.to(DEVICE))
        o, h = self.rnn(embed)
        return self.linear(o)


class RNNModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 32)
        self.rnn = torch.nn.RNN(32, 64, batch_first=True)
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, sentence, state=None):
        embed = self.embed(sentence.to(DEVICE))
        o, h = self.rnn(embed)
        return self.linear(o)


class GRUModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(10, 128)
        self.rnn = torch.nn.GRU(128, 256, batch_first=True)
        self.linear = torch.nn.Linear(256, 10)

    def forward(self, sentence, state=None):
        embed = self.embed(sentence.to(DEVICE))
        o, h = self.rnn(embed)
        return self.linear(o)


class Model(object):

    def __init__(self, model_name):
        self.model_name = model_name
        self.BATCH_SIZE = 10

    def sequence(self, x):
        y = []
        for i in range(len(x)):
            res = []
            for j in range(len(x[0])):
                if j == 0:
                    n = x[i][j]
                else:
                    n = x[i][0] + x[i][j]
                if n >= 10:
                    n = n - 10
                res.append(n)
            y.append(res)
        return y

    def dataset(self):
        x_train_sequence = [[random.randint(0, 9) for i in range(256)] for i in range(1000)]
        y_train_sequence = self.sequence(x_train_sequence)
        X = torch.tensor(x_train_sequence)
        Y = torch.tensor(y_train_sequence)

        dataset = torch.utils.data.TensorDataset(X, Y)
        train_dataset = torch.utils.data.DataLoader(dataset, self.BATCH_SIZE, shuffle=True)

        x_test_sequence = [[random.randint(0, 9) for i in range(256)] for i in range(200)]
        y_test_sequence = self.sequence(x_test_sequence)
        X_test = torch.tensor(x_test_sequence)
        Y_test = torch.tensor(y_test_sequence)

        test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

        return train_dataset, test_dataset

    def ModelTrain(self):
        train_dataset, test_dataset = self.dataset()

        if self.model_name == "RNN":
            model = RNNModel().to(DEVICE)
            num_epochs = 600
        elif self.model_name == "LSTM":
            model = LSTMModel().to(DEVICE)
            num_epochs = 50
        elif self.model_name == "GRU":
            model = GRUModel().to(DEVICE)
            num_epochs = 15

        loss = torch.nn.CrossEntropyLoss().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(num_epochs):
            train_loss, train_acc, iter_num = .0, .0, .0
            start_epoch_time = time.time()
            model.train()
            for x_in, y_in in train_dataset:
                x_in = x_in.to(DEVICE)
                y_in = y_in.to(DEVICE).view(1, -1).squeeze()
                optimizer.zero_grad()
                out = model.forward(x_in).view(-1, 10)
                l = loss(out, y_in)
                train_loss += l.item()
                batch_acc = (out.argmax(dim=1) == y_in)
                train_acc += batch_acc.sum().item() / batch_acc.shape[0]
                l.backward()
                optimizer.step()
                iter_num += 1
            print(
                f"Epoch: {epoch}, loss: {train_loss:.4f}, acc: "
                f"{train_acc / iter_num:.4f}",
                end=" | ",
            )

            test_loss, test_acc, iter_num = .0, .0, .0
            model.eval()
            for x_in, y_in in test_dataset:
                x_in = x_in.to(DEVICE)
                y_in = y_in.to(DEVICE).view(1, -1).squeeze()
                out = model.forward(x_in).view(-1, 10)
                l = loss(out, y_in)
                test_loss += l.item()
                batch_acc = (out.argmax(dim=1) == y_in)
                test_acc += batch_acc.sum().item() / batch_acc.shape[0]
                iter_num += 1
            print(
                f"test loss: {test_loss:.4f}, test acc: {test_acc / iter_num:.4f} | "
                f"{time.time() - start_epoch_time:.2f} sec."
            )

        x_val = [random.randint(0, 9) for i in range(256)]
        print("Random X sequence: ", torch.tensor(x_val))
        y_val = torch.tensor(self.sequence([x_val]))
        print("Real Y sequence: ", y_val)
        y_pred = model(torch.tensor(x_val)).to(DEVICE).argmax(dim=1)
        print("Predicted Y sequence: ", y_pred)


if __name__ == '__main__':
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])

    model = namespace.model

    translator = Model(model)
    translator.ModelTrain()
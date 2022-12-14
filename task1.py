
import math
from random import shuffle
import time
import re

import torch


def DataPreprocess(txt_path):
    with open(txt_path, encoding='utf-8') as txt_file:
        text = txt_file.read()
    text = re.sub(r"[^а-яА-Я0-9.!?,:—()-]+", r" ", text)
    text = text.replace('ё', 'е')
    text = re.sub('\s+', ' ', text)
    txt_file.close()
    return text


class LettersList(object):

    def __init__(self):
        self.letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789"

    def __len__(self):
        return len(self.letters)

    def __contains__(self, item):
        return item in self.letters

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.letters[item % len(self.letters)]
        elif isinstance(item, str):
            return self.letters.find(item)

    def load_from_file(self, file_path):
        text = DataPreprocess(file_path)
        for ch in text:
            if ch not in self.letters:
                self.letters += ch
        return self


class Encrypt(object):

    def encrypt(self, input_text, key):
        res, n = [], ""
        dictionary, dictionary_upper = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

        for i in range(len(input_text)):
            if input_text[i] in dictionary:
                n = dictionary
            elif input_text[i] in dictionary_upper:
                n = dictionary_upper
            else:
                res.append(input_text[i])

            if input_text[i] in n:
                for j in range(len(n)):
                    if 0 <= j + key < len(n) and input_text[i] == n[j]:
                        res.append(n[j + key])
                    elif j + key >= len(n) and input_text[i] == n[j]:
                        res.append(n[(1 - j - key) % (len(n) - 1)])
                    elif j + key < 0 and input_text[i] == n[j]:
                        res.append(n[(j + key) % len(n)])

            encrypted = ''.join(res)

        return encrypted


class SentenceDataset(torch.utils.data.Dataset):

    def __init__(self, raw_data, letters, key):
        super().__init__()
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._len = len(raw_data)
        self.y = torch.tensor(
            [[letters[ch] for ch in line] for line in raw_data]
        ).to(self.DEVICE)
        self.x_str = [Encrypt().encrypt(line, key) for line in raw_data]
        self.x = torch.tensor([[letters[ch] for ch in line] for line in self.x_str]).to(self.DEVICE)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RNNModel(torch.nn.Module):

    def __init__(self, letters_list, offset):
        super().__init__()
        self.embed = torch.nn.Embedding(len(letters_list) + offset, 32)
        self.rnn = torch.nn.RNN(32, 128, batch_first=True)
        self.linear = torch.nn.Linear(128, len(letters_list) + offset)
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, sentence, state=None):
        embed = self.embed(sentence.to(self.DEVICE))
        o, h = self.rnn(embed)
        return self.linear(o)


class CAESAR(object):
    def __init__(self, train_file, offset):
        self.BATCH_SIZE = 10
        self.STRING_SIZE = 60
        self.train_file = train_file
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.offset = offset
        self.text = DataPreprocess(self.train_file)
        self.letters_list = LettersList().load_from_file(self.train_file)

    def get_text_array(self, text):
        MAX_LEN = 100
        STEP = 40
        SENTENCES = []

        for i in range(0, len(text) - MAX_LEN, STEP):
            SENTENCES.append(text[i: i + MAX_LEN])
        return SENTENCES

    def get_dataset(self):
        raw_data = self.get_text_array(self.text)

        shuffle(raw_data)
        raw_data = raw_data[math.ceil(len(raw_data) * 0.1):]
        test_data = raw_data[:math.ceil(len(raw_data) * 0.2)]
        train_data = raw_data[math.ceil(len(raw_data) * 0.2):]

        train = torch.utils.data.DataLoader(
            SentenceDataset(
                train_data, self.letters_list, self.offset
            ),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        test = torch.utils.data.DataLoader(
            SentenceDataset(
                test_data, self.letters_list, self.offset
            ),
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        return train, test

    def model(self, TEST_FILE_NAME):
        model = RNNModel(self.letters_list, self.offset).to(self.DEVICE)
        loss = torch.nn.CrossEntropyLoss().to(self.DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        train, test = self.get_dataset()

        for epoch in range(50):
            train_loss, train_acc, iter_num = .0, .0, .0
            start_epoch_time = time.time()
            model.train()
            for x_in, y_in in train:
                x_in = x_in
                y_in = y_in.view(1, -1).squeeze()
                optimizer.zero_grad()
                out = model.forward(x_in).view(-1, len(self.letters_list) + self.offset)
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
                end=" | "
            )
            test_loss, test_acc, iter_num = .0, .0, .0
            model.eval()
            for x_in, y_in in test:
                x_in = x_in
                y_in = y_in.view(1, -1).squeeze()
                out = model.forward(x_in).view(-1, len(self.letters_list) + self.offset)
                l = loss(out, y_in)
                test_loss += l.item()
                batch_acc = (out.argmax(dim=1) == y_in)
                test_acc += batch_acc.sum().item() / batch_acc.shape[0]
                iter_num += 1
            print(
                f"test loss: {test_loss:.4f}, test acc: {test_acc / iter_num:.4f} | "
                f"{time.time() - start_epoch_time:.2f} sec."
            )

        sentence = DataPreprocess(TEST_FILE_NAME)
        sentence_idx = torch.tensor([self.letters_list[ch] for ch in sentence]).to(self.DEVICE)
        encrypted_sentence_idx = [self.letters_list[i] for i in Encrypt().encrypt(sentence, self.offset)]
        encrypted_sentence = "".join([self.letters_list[i] for i in encrypted_sentence_idx])
        result = model(torch.tensor([encrypted_sentence_idx])).to(self.DEVICE).argmax(dim=2)
        val_acc = (result == sentence_idx.to(self.DEVICE)).flatten()
        val_acc = (val_acc.sum() / val_acc.shape[0]).item()
        deencrypted_sentence = "".join([self.letters_list[i.item()] for i in result.flatten()])
        print(f"Encrypted sentence is : {encrypted_sentence}")
        print(f"Deencrypted sentence is : {deencrypted_sentence}")
        print(f"Accuracy: {val_acc:.4f}")


if __name__ == '__main__':

    caeser_cipher = CAESAR('data_task_1/onegin.txt', 2)
    caeser_cipher.model('data_task_1/test.txt')
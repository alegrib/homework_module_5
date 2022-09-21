from io import open
import unicodedata
import time
import math
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class LanguageVocabulary(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)


    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataPreprocess(object):

    def __init__(self, lang1, lang2):
        self.lang1 = lang1
        self.lang2 = lang2

    def unicode_to_ascii(self, s):
        return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Zа-яА-ЯІіў.,!?–:—’']+", r" ", s)
        return s

    def read_languages(self, reverse=False):
        lines = open('translation_data/%s-%s.txt' % (self.lang1, self.lang2), encoding='utf-8').read().strip().split('\n')
        pairs = [[self.normalize_string(s) for s in l.split('\t')] for l in lines]

        for i in range(len(pairs)):
            pairs[i].pop(2)

        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = LanguageVocabulary(self.lang2)
            output_lang = LanguageVocabulary(self.lang1)
        else:
            input_lang = LanguageVocabulary(self.lang1)
            output_lang = LanguageVocabulary(self.lang2)
        return input_lang, output_lang, pairs

    def prepare_data(self, reverse=False):
        input_lang, output_lang, pairs = self.read_languages(reverse)
        for pair in pairs:
            input_lang.add_sentence(pair[0])
            output_lang.add_sentence(pair[1])
        return input_lang, output_lang, pairs

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, pair, input_lang, output_lang):
        input_tensor = self.tensorFromSentence(input_lang, pair[0])
        target_tensor = self.tensorFromSentence(output_lang, pair[1])
        return (input_tensor, target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class TranslatorModel(object):

    def __init__(self):
        super(TranslatorModel, self).__init__()
        self.pre_data = DataPreprocess('eng', 'bel')
        self.input_lang, self.output_lang, self.pairs = self.pre_data.prepare_data(True)
        self.teacher_forcing_ratio = 0.5


    def maxlen(self, pairs):
        len_list = []

        for i in range(len(pairs)):
            for j in range(len(pairs[i])):
                len_list.append(len(pairs[i][j]))

        MAX_LENGTH = max(len_list)

        return MAX_LENGTH

    def train(self, input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
        max_length = self.maxlen(self.pairs)
        encoder_hidden = encoder.initHidden()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        loss = 0
        train_acc_sum = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                loss += criterion(decoder_output, target_tensor[di])
                batch_acc = (decoder_output.argmax(dim=1) == target_tensor)
                train_acc_sum += batch_acc.sum().item() / batch_acc.shape[0]
                decoder_input = target_tensor[di]
        else:
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                loss += criterion(decoder_output, target_tensor[di])
                batch_acc = (decoder_output.argmax(dim=1) == target_tensor)
                train_acc_sum += batch_acc.sum().item() / batch_acc.shape[0]
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        return loss.item() / target_length, train_acc_sum / target_length

    def asMinutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(self, since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- eta: %s)' % (self.asMinutes(s), self.asMinutes(rs))

    def trainIters(self, encoder, decoder, n_iters, print_every=1000, learning_rate=0.01):
        start = time.time()
        print_loss_total = 0
        train_acc_sum = .0

        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [self.pre_data.tensorsFromPair(random.choice(self.pairs), self.input_lang, self.output_lang) for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for epoch in range(1, n_iters + 1):
            training_pair = training_pairs[epoch - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]
            loss, accuracy = self.train(input_tensor, target_tensor, encoder,
                                   decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            train_acc_sum += accuracy

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                train_acc_sum_avg = train_acc_sum / print_every
                print_loss_total = 0
                print('%s (%d %d%%) Loss: %.4f Accuracy: %.4f' % (self.timeSince(start, epoch / n_iters),
                                                                  epoch, epoch / n_iters * 100, print_loss_avg,
                                                                  train_acc_sum_avg))

    def evaluate(self, encoder, decoder, sentence):
        max_length = self.maxlen(self.pairs)
        with torch.no_grad():
            input_tensor = self.pre_data.tensorFromSentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = encoder.initHidden()
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            for i in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
                encoder_outputs[i] += encoder_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            decoded_words = []

            for di in range(max_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])
                decoder_input = topi.squeeze().detach()
        return decoded_words


    def evaluateRandomly(self, encoder, decoder, n=10):
        for i in range(n):
            pair = random.choice(self.pairs)
            print('Input sentence in belarusian language: ', pair[0])
            print('Real sentence: ', pair[1])
            val_pairs = self.pre_data.tensorsFromPair(pair, self.input_lang, self.output_lang)
            output_words = self.evaluate(encoder, decoder, pair[0])
            output_sentence = ' '.join(output_words)
            indexes = torch.tensor([self.output_lang.word2index[word] for word in output_sentence.replace(' <EOS>', '').split(' ')], dtype=torch.long,device=device).view(-1, 1)
            k = val_pairs[1][:val_pairs[1].size(0) - 1]
            indexes = F.pad(indexes, pad=(0, 0, 0, max(indexes.size(0), k.size(0)) - indexes.size(0)), mode='constant', value=0)
            val_pairs_pad = F.pad(k, pad=(0, 0, 0, max(indexes.size(0), k.size(0)) - k.size(0)), mode='constant', value=0)
            print('Translated sentence: ', output_sentence)
            print('')
            val_acc = (indexes == val_pairs_pad).flatten()
            val_acc = (val_acc.sum() / val_acc.shape[0]).item()
            print(f"Accuracy: {val_acc:.4f}")

    def modeltesting(self):
        encoder1 = EncoderRNN(self.input_lang.n_words, 256).to(device)
        decoder1 = DecoderRNN(256, self.output_lang.n_words).to(device)
        self.trainIters(encoder1, decoder1, 44500, print_every=5500)
        self.evaluateRandomly(encoder1, decoder1)

if __name__ == '__main__':

    translator = TranslatorModel()
    translator.modeltesting()

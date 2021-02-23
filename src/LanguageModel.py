import os
import random
import string

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

PAD = "@@PAD@@"
UNK = "@@UNK@@"

EMBEDDING_DIM = 16
BATCH_SIZE = 32
HIDDEN_DIM = 32
N_RNN_LAYERS = 2
LEARNING_RATE = 1e-3

USE_LSTM = True

EPOCH = 20

CUT = 128

TRAINING_PATH = "work_dir/training/1b_benchmark.train.tokens"
CHECKPOINT = 'model.checkpoint2'


class LMDataset(Dataset):
    """
    create dataset
    """

    def __init__(self, dataset):
        """
        tensorize dataset
        :param dataset:
        """

        def tensorize(dataset):
            """

            :param dataset: List[str], len(str)=32,
             x-> len(dataset) * 31的tensor（int），y->len(dataset)的tensor（int)
            :return:
            """
            x_value = [torch.tensor(line[:CUT - 1], dtype=torch.long) for line in dataset]
            y_value = [torch.tensor(line[-1], dtype=torch.long) for line in dataset]
            return torch.stack(x_value, dim=0), torch.stack(y_value, dim=0)

        self.x, self.y = tensorize(dataset)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class RNNModel(nn.Module):
    """WEI"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, device):
        """
        :param vocab_size: vocabulary size
        :param embedding_dim: embedding dimension
        :param hidden_dim: hidden dimension
        :param n_labels: number of labels
        :param n_rnn_layers: number of rnn layers
        :param drop_rate: dropout rate
        """
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        # drop_rate = 0.5
        # self.dropout = nn.Dropout(drop_rate)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_rnn_layers, batch_first=True, bidirectional=False)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_rnn_layers, batch_first=True, bidirectional=True)
        layered_hidden_dim = hidden_dim * n_rnn_layers * 2
        self.output = nn.Linear(layered_hidden_dim, n_labels)

        self.linear = nn.Linear(hidden_dim, n_labels)
        self.activation = nn.Softmax()

        hidden_state = Variable(torch.randn(n_rnn_layers, BATCH_SIZE, hidden_dim).to(device))
        cell_state = Variable(torch.randn(n_rnn_layers, BATCH_SIZE, hidden_dim).to(device))
        self.hidden = (hidden_state, cell_state)

    def forward(self, data):
        """
        :param data:
        :return:
        """
        if USE_LSTM:
            # embeds = self.dropout(embeds)
            embeds = self.encoder(data)
            output, hiddenx = self.lstm(embeds, None)

            linear_output = self.linear(output[:, -1, :])

            # output = self.activation(output)
            return linear_output
        else:
            embeds = self.encoder(data)
            _, hidden = self.gru(embeds)
            hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)
            return self.output(hidden)



class LanguageModel:

    def __init__(self, **kwargs):
        if 'saved' in kwargs and 'model_state_dict' in kwargs:
            print('Loading model from data')
            saved = kwargs['saved']
            model_state_dict = kwargs['model_state_dict']
            self.character_to_idx = saved['character_to_idx']
            self.idx_to_character = saved['idx_to_character']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = RNNModel(
                len(self.character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(self.character_to_idx), N_RNN_LAYERS, self.device
            )
            self.model.load_state_dict(model_state_dict)
            self.model = self.model.to(self.device)
        else:
            print('Constructing a new model')
            self.construct()

    def construct(self):
        path = TRAINING_PATH
        train_data = LanguageModel.load_training_data(path)

        self.character_to_idx, self.idx_to_character = self.create_vocab(train_data)

        train_data = self.apply_vocab(train_data, self.character_to_idx)
        # apply_label(train_data, character_to_idx)  # 每一行最后一个字母就是label

        train_dataset = LMDataset(train_data)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        """YiWEN"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RNNModel(
            len(self.character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(self.character_to_idx), N_RNN_LAYERS, self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.model = self.model.to(self.device)

    def apply_vocab(self, train_data, character_to_idx):
        """
        map character to int
        :param train_data: List[str] -> map to -> List[List[int]]
        :param character_to_idx: dict[str, int]
        :return: List[List[int]]
        """
        idx_data = []
        for line in train_data:
            idx_line = []
            for j in range(len(line)):
                idx_line.append(character_to_idx.get(line[j]) or character_to_idx[UNK])
            idx_data.append(idx_line)
        return idx_data

    def create_vocab(self, train_data):
        """

        :param train_data: List[str]
        :return:
        character_to_idx: dict[Char, Int],
        idx_to_character: dict[Int, Char]
        """
        character_to_idx = {}
        idx_to_character = {}
        character_to_idx["@@PAD@@"] = 0
        character_to_idx["@@UNK@@"] = 1
        idx_to_character[0] = "@@PAD@@"
        idx_to_character[1] = "@@UNK@@"
        i = 2
        for line in train_data:
            for j in range(len(line)):
                if line[j] not in character_to_idx.keys():
                    character_to_idx[line[j]] = i
                    idx_to_character[i] = line[j]
                    i += 1
        return character_to_idx, idx_to_character

    @classmethod
    def load_training_data(cls, path=TRAINING_PATH):
        """
        裁剪成32长度的str，不够的ignore
        read from training file, return
        :return: List[str] Each string is a sample
        """
        # your code here
        # this particular model doesn't train
        lines = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) >= CUT:
                    line = line[:CUT]
                    lines.append(line)
        return lines

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname, encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self):
        # your code here
        def train(model, train_dataloader, optimizer, device):
            """YUAN"""
            model.train()
            for texts, labels in tqdm(train_dataloader):
                texts, labels = texts.to(device), labels.to(device)
                # print(texts)
                output = model(texts)
                loss = F.cross_entropy(output, labels)
                model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            print(f'loss: {loss}')

        for epoch in range(EPOCH):
            train(self.model, self.train_dataloader, self.optimizer, self.device)


    def run_pred(self, data):
        # your code here
        """YUAN"""

        def tensorize(test_data):
            """
            Tensorize the input data -> convert to list of tensors
            :param test_data: List[List[int]]
            :return: List[Tensor[int]]
            """
            res = []
            for input in data:
                res.append(torch.tensor(input).to(self.device))
            return res

        data = self.apply_vocab(data, self.character_to_idx)
        data = tensorize(data)

        preds = []
        with torch.no_grad():
            for input in data:
                output = self.model(input.reshape(1, -1)).reshape(-1)
                predicted_idx = output.argsort(dim=-1, descending=True)[:5]
                predicted_idx = list(filter(lambda idx: self.idx_to_character[idx.item()] != UNK
                                                        and self.idx_to_character[idx.item()] != PAD, predicted_idx))[:3]
                predicted_char = [self.idx_to_character[idx.item()] for idx in predicted_idx]
                preds.append(''.join(predicted_char))

        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        save = dict(
            torch_model_state_dict=self.model.state_dict(),
            lm_dict=dict(
                device=self.device,
                character_to_idx=self.character_to_idx,
                idx_to_character=self.idx_to_character
            )
        )
        torch.save(save, os.path.join(work_dir, CHECKPOINT))

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        save = torch.load(os.path.join(work_dir, CHECKPOINT))
        return LanguageModel(saved=save['lm_dict'], model_state_dict=save['torch_model_state_dict'])

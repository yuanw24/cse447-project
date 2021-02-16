import os
import random
import string

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

PAD = "@@PAD@@"
UNK = "@@UNK@@"

EMBEDDING_DIM = 32
BATCH_SIZE = 32
HIDDEN_DIM = 32
N_RNN_LAYERS = 2
LEARNING_RATE = 1e-3

CUT = 32


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
            x_value = [torch.tensor(line[:31], dtype=torch.long) for line in dataset]
            y_value = [torch.tensor(line[-1], dtype=torch.long) for line in dataset]
            return torch.stack(x_value, dim=0), torch.stack(y_value, dim=0)

        self.x, self.y = tensorize(dataset)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class RNNModel(nn.Module):
    """WEI"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, pad_idx):
        super().__init__()

    def forward(self, data):
        pass


class LanguageModel:

    def __init__(self):
        path = "../work_dir/training/1b_benchmark.train.tokens"
        train_data = LanguageModel.load_training_data(path)

        self.character_to_idx, self.idx_to_character = self.create_vocab(train_data)

        train_data = self.apply_vocab(train_data, self.character_to_idx)
        # apply_label(train_data, character_to_idx)  # 每一行最后一个字母就是label

        train_dataset = LMDataset(train_data)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        """YiWEN"""

        self.model = RNNModel(
            len(self.character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(self.character_to_idx), N_RNN_LAYERS
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                idx_line.append(character_to_idx[line[j]])
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
    def load_training_data(cls):
        """
        裁剪成32长度的str，不够的ignore
        read from training file, return
        :return: List[str] Each string is a sample
        """
        # your code here
        # this particular model doesn't train
        f = open(cls, "r")
        lines = []
        while True:
            line = f.readline()
            if not line:
                break
            else:
                if len(line) >= 32:
                    line = line[:32]
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
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        def train(model, train_dataloader, optimizer, device):
            """YUAN"""
            for texts, labels in tqdm(train_dataloader):
                texts, labels = texts.to(device), labels.to(device)
                output = model(texts)
                loss = F.cross_entropy(output, labels)
                model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        train(self.model, self.train_dataloader, self.optimizer, self.device)
        pass

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
        for input in data:
            output = self.model(input)
            predicted_idx = output.argsort(dim=-1)[:3]
            predicted_char = [self.idx_to_character[idx] for idx in predicted_idx]
            preds.append(''.join(predicted_char))

        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return LanguageModel()

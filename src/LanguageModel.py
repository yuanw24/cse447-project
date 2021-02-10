import os
import string
import random

import torch
from torch.utils.data import Dataset, DataLoader


PAD = "@@PAD@@"
UNK = "@@UNK@@"

EMBEDDING_DIM = ""

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
            pass
        self.x, self.y = tensorize(dataset)



    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class RNNModel(nn.Module):
    """WEI"""
    pass


class LanguageModel:

    def __init__(self):
        train_data = LanguageModel.load_training_data()

        character_to_idx, idx_to_character = create_vocab(train_data)

        apply_vocab(train_data, character_to_idx)
        apply_label(train_data, character_to_idx)  # 每一行最后一个字母就是label

        train_dataset = LMDataset(train_data)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        """YiWEN"""

        self.model = RNNModel(
            len(character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(character_to_idx), N_RNN_LAYERS
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize(self):


    def create_vocab(self, train_data):
        """

        :param train_data: List[str]
        :return:
        character_to_idx: dict[Char, Int],
        idx_to_character: dict[Int, Char]
        """

    @classmethod
    def load_training_data(cls):
        """
        裁剪成32长度的str，不够的ignore
        read from training file, return
        :return: List[str] Each string is a sample
        """
        # your code here
        # this particular model doesn't train
        return []

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
            pass
        train(model, train_dataloader, optimizer, device)
        pass

    def run_pred(self, data):
        # your code here
        def evaluate(model, test_dataloader, device):
            """YUAN"""
            pass
        evaluate(model, test_dataloader, device)

        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
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
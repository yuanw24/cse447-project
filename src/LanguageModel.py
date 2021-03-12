import os
import random
import string
from collections import Counter

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from tqdm import tqdm

PAD = "@@PAD@@"
UNK = "@@UNK@@"
PAD_IDX = 0
UNK_IDX = 1
UNK_LIMIT = 5

EMBEDDING_DIM = 16
BATCH_SIZE = 512
HIDDEN_DIM = 64
N_RNN_LAYERS = 2
LEARNING_RATE = 1e-1

USE_LSTM = True
DEBUGGING = False

EPOCH = 5

CUT = 64

# TRAINING_PATH = "work_dir/training/1b_benchmark.train.tokens"
# LANGS_CUT = {}
TRAINING_PATH = "work_dir/training-monolingual/{}.filtered"
LANGS_CUT = {
    'en': 64,
    'cs': 64,
    'de': 64,
    'fr': 64,
    'es': 64,
    'cn': 32,
    # 'cn-tr': 16,
    'jp': 16,
    'ru': 16,
}
CHECKPOINT = 'model.checkpoint4.cut32-16'


class LMDataset(Dataset):
    """
    create dataset
    """

    def __init__(self, dataset, pad_idx, sort=True, ignoreLast=True):
        """
        tensorize dataset
        :param dataset: List[List[int]]
        """
        if sort:
            dataset = sorted(dataset, key=lambda data: len(data))

        if ignoreLast:
            self.text = [inst[:-1] for inst in dataset]
        else:
            self.text = dataset

        self.label = [inst[-1] for inst in dataset]
        self.pad_idx = pad_idx

    def __getitem__(self, idx):
        return self.text[idx], self.label[idx]

    def __len__(self):
        return len(self.text)

    def collate_fn(self, batch):
        def tensorize(elements, dtype):
            return [torch.tensor(element, dtype=dtype) for element in elements]

        def pad(tensors):
            """Assumes 1-d tensors."""
            max_len = max(len(tensor) for tensor in tensors)
            padded_tensors = [
                F.pad(tensor, (0, max_len - len(tensor)), value=self.pad_idx) for tensor in tensors
            ]
            return padded_tensors

        texts, labels = zip(*batch)
        return [
            torch.stack(pad(tensorize(texts, torch.long)), dim=0),
            torch.stack(tensorize(labels, torch.long), dim=0),
        ]


class RNNModel(nn.Module):
    """WEI"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_labels, n_rnn_layers, device):
        """
        :param vocab_size: vocabulary size
        :param embedding_dim: embedding dimension
        :param hidden_dim: hidden dimension
        :param n_labels: number of labels
        :param n_rnn_layers: number of rnn layers
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
        non_padded_positions = data != PAD_IDX
        lens = non_padded_positions.sum(dim=1)
        embedded = self.encoder(data)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lens.cpu(), batch_first=True, enforce_sorted=False
        )

        if USE_LSTM:
            output, (h_n, h_c) = self.lstm(packed_embedded, None)
            linear_output = self.linear(h_n[-1, :, :])
            # output = self.activation(output)
            return linear_output
        else:
            _, hidden = self.gru(embedded)
            hidden = hidden.transpose(0, 1).reshape(hidden.shape[1], -1)
            return self.output(hidden)


class LanguageModel:

    def __init__(self, **kwargs):
        self.character_to_idx = None
        self.idx_to_character = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_dataloader = None
        self.optimizer = None

        if 'saved' in kwargs and 'model_state_dict' in kwargs:
            print('Loading model from data')
            saved = kwargs['saved']
            model_state_dict = kwargs['model_state_dict']
            self.character_to_idx = saved['character_to_idx']
            self.idx_to_character = saved['idx_to_character']
            self.device = torch.device("cpu")
            model = RNNModel(
                len(self.character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(self.character_to_idx), N_RNN_LAYERS,
                self.device
            )
            model.load_state_dict(model_state_dict)
            self.model = model.to(self.device)
        else:
            print('Constructing a new model')
            self.construct()

    def construct(self):
        path = TRAINING_PATH
        # print('Load Training Data')
        train_data = LanguageModel.load_training_data(path)
        self.character_to_idx, self.idx_to_character = self.create_vocab(train_data)
        train_data = self.apply_vocab(train_data, self.character_to_idx)

        train_dataset = LMDataset(train_data, pad_idx=PAD_IDX)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn
        )

        """YiWEN"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNNModel(
            len(self.character_to_idx), EMBEDDING_DIM, HIDDEN_DIM, len(self.character_to_idx), N_RNN_LAYERS, self.device
        ).to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    def apply_vocab(self, train_data, character_to_idx):
        """
        map character to int
        :param train_data: List[str] -> map to -> List[List[int]]
        :param character_to_idx: dict[str, int]
        :return: List[List[int]]
        """
        print('Applying Vocab')
        idx_data = []
        for line in tqdm(train_data):
            idx_line = []
            for j in range(len(line)):
                idx_line.append(character_to_idx.get(line[j]) or UNK_IDX)
            idx_data.append(idx_line)
        return idx_data

    def create_vocab(self, train_data):
        """

        :param train_data: List[str]
        :return:
        character_to_idx: dict[Char, Int],
        idx_to_character: dict[Int, Char]
        """
        print('Creating Vocab')
        character_to_idx = {}
        idx_to_character = {}
        character_to_idx[PAD] = PAD_IDX
        character_to_idx[UNK] = PAD_IDX
        idx_to_character[PAD_IDX] = PAD
        idx_to_character[UNK_IDX] = UNK
        i = 2

        counter = Counter()

        for line in tqdm(train_data):
            for j in range(len(line)):
                if line[j] not in character_to_idx:
                    if line[j] in counter and counter.get(line[j]) >= UNK_LIMIT:
                        character_to_idx[line[j]] = i
                        idx_to_character[i] = line[j]
                        i += 1
                        counter.pop(line[j])
                    else:
                        counter.update(line[j])

        # for k, v in counter.items():
        #     if v >= UNK_LIMIT:
        #         character_to_idx[k] = i
        #         idx_to_character[i] = k
        #         i += 1
        #
        # for line in tqdm(train_data):
        #     for j in range(len(line)):
        #         if line[j] not in character_to_idx.keys():
        #             character_to_idx[line[j]] = i
        #             idx_to_character[i] = line[j]
        #             i += 1
        return character_to_idx, idx_to_character

    @classmethod
    def load_training_data(cls, path=TRAINING_PATH):
        """
        read from training file
        :return: List[str] Each string is a sample
        """
        # your code here
        # this particular model doesn't train
        lines = []
        if LANGS_CUT:
            for lang in LANGS_CUT:
                with open(path.format(lang), 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) >= LANGS_CUT[lang]:
                            line = line[:LANGS_CUT[lang]]
                            lines.append(line)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
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
        if DEBUGGING:
            data = data[:10000]

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
            total_loss = 0
            batch_idx = 1
            model.train()
            for texts, labels in tqdm(train_dataloader):
                optimizer.zero_grad()
                texts, labels = texts.to(device), labels.to(device)
                # print(texts)
                output = model(texts)
                loss = F.cross_entropy(output, labels)
                # model.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                total_loss += loss.item()
                batch_idx += 1
            print(f'Total loss: {total_loss / batch_idx}')

        for epoch in range(EPOCH):
            print(f'Epoch: {epoch}')
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
            for input in test_data:
                res.append(torch.tensor(input).to(self.device))
            return res

        data = self.apply_vocab(data, self.character_to_idx)
        dataset = LMDataset(data, pad_idx=PAD_IDX, sort=False, ignoreLast=False)
        data_loader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)

        preds = []
        with torch.no_grad():
            for input, _ in tqdm(data_loader):
                input = input.to(self.device)
                outputs = self.model(input)
                for output in outputs:
                    predicted_idx = output.argsort(dim=-1, descending=True)[:5]
                    predicted_idx = list(filter(lambda idx: self.idx_to_character[idx.item()] != UNK
                                                            and self.idx_to_character[idx.item()] != PAD,
                                                predicted_idx))[
                                    :3]
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
        save = torch.load(os.path.join(work_dir, CHECKPOINT), map_location=torch.device('cpu'))
        return LanguageModel(saved=save['lm_dict'], model_state_dict=save['torch_model_state_dict'])

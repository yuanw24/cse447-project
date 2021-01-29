# -*- coding: utf-8 -*-

"""
A character-level trie model used as a language model.
Returns the three most-likely character after the given prefix of the word.
If an empty string is given, return the most-likely character.
If not enough answers can be found at the given node, fill the answer to 3
with search from the root.

2021/1/29
"""

import os
import sys
import pickle


class TrieModel():
    """
    Implements all parts of MyModel using a Trie schema.
    A character-level trie model used as a language model.

    p(`|prefix) = c(prefix-`) / c(prefix)

    Answer is chosen as the three most frequent characters after the node.

    root:  the root node.
    """

    training_file = 'work_dir/training/1b_benchmark.train.tokens'

    def __init__(self):
        self.root = Node()

    @classmethod
    def load_training_data(cls):
        """
        Loads training data from TrieModel.training_file, split by line, then by whitespace
        :return: List[List[str]], training data
        """
        res = []
        with open(TrieModel.training_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                res.append(line.split())
        return res

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'w', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        """
        Trains the model by constructing the tree on every word from the training data
        :param data: List[List[str]], parsed training data
        :param work_dir: irrelevant
        """
        for inst in data:
            for word in inst:
                self.root.add(word + ' ')

    def run_pred(self, data):
        """
        Estimate the next most-probable character from input data
        :param data: List[str], input
        :return: List[str], predictions, each prediction is consists 3 characters
        """
        preds = []
        for inst in data:
            target_node = self.root
            if not inst.endswith(' '):
                last_word = inst.split()[-1]
                target_node = self.root.traverse(last_word) or self.root
            res = target_node.get_three()
            if len(res) < 3:
                root_res = self.root.get_three()
                res.extend(root_res[:3 - len(res)])
            preds.append(''.join(res))

        return preds

    def save(self, work_dir):
        """
        Saves the model to work_dir/model.checkpoint
        :param work_dir: path to working directory
        """
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, work_dir):
        """
        Loads the model from work_dir/model.checkpoint
        :param work_dir: path to working directory
        :return the model loaded
        """
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            return pickle.load(f)


class Node:
    """
    The node of the trie.
    next:  dict[str, Node], the character to next node
    count: int, sum of all the child nodes' counts, the frequency of this particular node.
    """

    def __init__(self):
        self.next = {}
        self.count = 0

    def incr(self):
        self.count += 1

    def get_three(self):
        """
        :return: List[str], the three most probable character from the current node
        """
        bests = {}
        for k, v in self.next.items():
            bests[v.count] = k
            if len(bests) > 3:
                new_bests = {}
                c = 0
                for key in sorted(bests, reverse=True):
                    if c == 3:
                        break
                    new_bests[key] = bests[key]
                    c += 1
                bests = new_bests
        return list(bests.values())

    def add(self, str):
        """
        Add the str to the current node and subnodes
        :param str: the remaining str
        """
        self.incr()
        if not str:
            return
        if str[0] not in self.next:
            self.next[str[0]] = Node()
        self.next[str[0]].add(str[1:])

    def traverse(self, target):
        """
        Traverse the tree according to the target;
        :param target: str, the target to be traversed
        :return: Node, the target Node if found, return None otherwise
        """
        if not target:
            return self
        if target[0] not in self.next:
            return None

        return self.next[target[0]].traverse(target[1:])

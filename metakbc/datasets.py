# -*- coding: utf-8 -*-

import os
import torch
from torch import LongTensor, Tensor
from pathlib import Path
from typing import Dict, Tuple, List, Optional

DATA_PATH = Path(os.path.dirname(__file__)) / "data"
file_ext = ".tsv"


class Dataset(object):

    def __init__(self, name: str, splits: List[str] = None) -> None:

        self.name = name
        self.root = DATA_PATH / name
        self.splits = splits

        if self.splits == None:
            self.splits = [file_path.stem for file_path in self.root.glob('*' + file_ext)]
        if not ('train' in self.splits and 'valid' in self.splits and 'test' in self.splits):
            raise(ValueError("splits does not contain train, valid and test"))

        entities, predicates = set(), set()
        for split in self.splits:
            file_name = split + file_ext
            file_path = self.root / file_name
            with open(file_path, 'r') as to_read:
                for i, line in enumerate(to_read.readlines()):
                    try:
                        s, p, o = line.strip().split('\t')
                    except Exception as e:
                        raise(IOError("Error reading split {} at line {}: {}".format(split, i, line)))
                    entities.add(s)
                    entities.add(o)
                    predicates.add(p)

        self.entities = sorted(entities)
        self.predicates = sorted(predicates)
        self.entities_to_id = {x: i for (i, x) in enumerate(self.entities)}
        self.predicates_to_id = {x: i for (i, x) in enumerate(self.predicates)}
        self.n_predicates = len(self.predicates)
        self.n_entities = len(self.entities)

        self.data = {}
        for split in self.splits:
            file_name = split + file_ext
            file_path = self.root / file_name

            examples = []
            with open(file_path, 'r') as to_read:
                for line in to_read.readlines():
                    s, p, o = line.strip().split('\t')
                    examples.append([self.entities_to_id[s], self.predicates_to_id[p], self.entities_to_id[o]])

            self.data[split] = LongTensor(examples)

        print("\nDataset: {}".format(name))
        print("entities: {} | predicates: {}".format(self.n_entities, self.n_predicates))
        print("train: {} | valid: {} | test: {}\n".format(len(self.data['train']), len(self.data['valid']), len(self.data['test'])))

    def get_examples(self, split: str) -> LongTensor:
        return self.data[split]

    def get_shape(self) -> Tuple:
        return self.n_entities, self.n_predicates, self.n_entities

    def get_batches(self, split: str, batch_size: int, shuffle: bool = False) -> Tensor:
        examples = self.data[split]
        n_examples = examples.shape[0]
        if shuffle: examples = examples[torch.randperm(n_examples)]
        batch_begin = 0
        while batch_begin < n_examples:
            yield examples[batch_begin:batch_begin + batch_size]
            batch_begin += batch_size
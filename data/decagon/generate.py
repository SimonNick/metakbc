#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
from sklearn.model_selection import KFold

import gzip
import csv

from typing import List, Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def read_csv(path):
    res = []
    with gzip.open(path, 'rt') as f:
        reader = csv.reader(f)
        for row in reader:
            res += [row]
    return res


def main(argv):
    path_lst = [
        "./targets/bio-decagon-targets.csv.gz",
        "./effectcategories/bio-decagon-effectcategories.csv.gz",
        "./mono/bio-decagon-mono.csv.gz",
        "./targets-all/bio-decagon-targets-all.csv.gz",
        "./ppi/bio-decagon-ppi.csv.gz",
        "./combo/bio-decagon-combo.csv.gz"
    ]

    combo_triples = []
    ppi_triples = []
    targets_all_triples = []
    targets_triples = []
    mono_triples = []
    categories_triples = []

    for path in path_lst:
        rows = read_csv(path)
        if 'bio-decagon-combo.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'pse:{row[2]}', f'stitch:{row[1]}')
                combo_triples += [triple]
        elif 'bio-decagon-ppi.csv' in path:
            for row in rows[1:]:
                triple = (f'gene:{row[0]}', f'ppi', f'gene:{row[1]}')
                ppi_triples += [triple]
        elif 'bio-decagon-targets-all.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'target', f'gene:{row[1]}')
                targets_all_triples += [triple]
        elif 'bio-decagon-targets.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'target', f'gene:{row[1]}')
                targets_triples += [triple]
        elif 'bio-decagon-mono.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'se', f'{row[1]}')
                mono_triples += [triple]
        elif 'bio-decagon-effectcategories.csv' in path:
            for row in rows[1:]:
                category_name = row[2].replace(" ", "_")
                triple = (f'{row[0]}', f'category', f'{category_name}')
                categories_triples += [triple]
        else:
            assert False

    # triples = combo_triples + ppi_triples + targets_all_triples + targets_triples + mono_triples + categories_triples
    triples = combo_triples + ppi_triples + targets_triples + mono_triples + categories_triples

    # sanity check
    assert len(triples) == len(set(triples))

    entity_set = {s for s, _, _ in triples} | {o for _, _, o in triples}
    for s, p, o in triples:
        for e in [s, o]:
            if ':' in e:
                a, b = e.split(':')
                assert b not in entity_set

    rs = np.random.RandomState(0)

    def shuffle(_triples: List[Tuple[str, str, str]]):
        res = [_triples[i] for i in rs.permutation(len(_triples))]
        assert len(res) == len(_triples)
        return res

    # combo_triples = shuffle(combo_triples)
    # ppi_triples = shuffle(ppi_triples)
    # targets_all_triples = shuffle(targets_all_triples)
    # targets_triples = shuffle(targets_triples)
    # mono_triples = shuffle(mono_triples)
    # categories_triples = shuffle(categories_triples)

    ppi_target_triples = shuffle(ppi_triples + targets_triples)
    categories_triples = shuffle(categories_triples)
    combo_triples = shuffle(combo_triples)

    with open('ppi_targets.tsv', 'w') as f:
        f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in ppi_target_triples])

    with open('categories.tsv', 'w') as f:
        f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in categories_triples])

    with open('polypharmacy_se.tsv', 'w') as f:
        f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in combo_triples])

    kf = KFold(n_splits=5)

    for i, (train, test) in enumerate(kf.split(combo_triples)):
        train_triples = [combo_triples[i] for i in train]
        heldout_triples = [combo_triples[i] for i in test]

        cutpoint = len(heldout_triples) // 2
        dev_triples = heldout_triples[:cutpoint]
        test_triples = heldout_triples[cutpoint:]

        with open(f'folds/train_{i}.tsv', 'w') as f:
            f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in train_triples])

        with open(f'folds/dev_{i}.tsv', 'w') as f:
            f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in dev_triples])

        with open(f'folds/test_{i}.tsv', 'w') as f:
            f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in test_triples])


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])


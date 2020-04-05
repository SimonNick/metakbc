#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import gzip
import csv

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

    pse_triples = []
    triples = []

    for path in path_lst:
        rows = read_csv(path)
        if 'bio-decagon-combo.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'pse:{row[2]}', f'stitch:{row[1]}')

                triples += [triple]
                pse_triples += [triple]
        elif 'bio-decagon-ppi.csv' in path:
            for row in rows[1:]:
                triple = (f'gene:{row[0]}', f'ppi', f'gene:{row[1]}')

                triples += [triple]
                pse_triples += [triple]
        elif 'bio-decagon-targets-all.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'target', f'gene:{row[1]}')

                # pse_triples += [triple]
                # triples += [triple]
        elif 'bio-decagon-targets.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'target', f'gene:{row[1]}')

                # pse_triples += [triple]
                triples += [triple]
        elif 'bio-decagon-mono.csv' in path:
            for row in rows[1:]:
                triple = (f'stitch:{row[0]}', f'se', f'{row[1]}')

                # pse_triples += [triple]
                triples += [triple]
        elif 'bio-decagon-effectcategories.csv' in path:
            for row in rows[1:]:
                category_name = row[2].replace(" ", "_")
                triple = (f'{row[0]}', f'category', f'{category_name}')

                # pse_triples += [triple]
                triples += [triple]
        else:
            assert False

    # sanity check
    entity_set = {s for s, _, _ in triples} | {o for _, _, o in triples}
    for s, p, o in triples:
        for e in [s, o]:
            if ':' in e:
                a, b = e.split(':')
                assert b not in entity_set

    with open('polypharmacy_pse.tsv', 'w') as f:
        f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in pse_triples])

    with open('polypharmacy_all.tsv', 'w') as f:
        f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in triples])


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])

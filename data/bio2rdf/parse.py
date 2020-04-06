#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import rdflib
import gzip

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    return _open(file, *args, **kwargs)


def main(argv):
    for path in argv:
        logger.info(f'Processing {path} ..')

        g = rdflib.ConjunctiveGraph()

        try:
            with iopen(path, 'rb') as f:
                g.parse(f, format='nquads', ignore_errors=True)
            new_path = path.replace(".nq.gz", ".tsv").replace(".nq", ".tsv")
            if path == new_path:
                new_path = f'{path}.tsv'
            with open(new_path, 'w') as f:
                f.writelines([f'{s}\t{p}\t{o}\n' for s, p, o in g])
            g.close()

        except:
            logger.error(f"Unexpected error: {sys.exc_info()}")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])

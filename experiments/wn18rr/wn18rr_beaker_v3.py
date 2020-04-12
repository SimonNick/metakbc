#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    mask_str = "--LM" if c["mvl"] else ""
    command = f'PYTHONPATH=. python3 ./bin/meta-cli.py ' \
        f'--train data/wn18rr/train.tsv ' \
        f'--dev data/wn18rr/dev.tsv ' \
        f'--test data/wn18rr/test.tsv ' \
        f'-m {c["m"]} -k {c["k"]} -b {c["b"]} -e {c["e"]} ' \
        f'-R {c["reg"]} -l {c["lr"]} --LL {c["llr"]} -W {c["ct"]} {mask_str} ' \
        f'-S {c["ls"]} -I {c["i"]} -V {c["V"]} --LE {c["LE"]} -o {c["o"]} -q '
    return command


def to_logfile(c, path):
    outfile = "{}/wn18rr_beaker_v3.{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        m=['complex'],
        k=[1000],
        b=[100],
        e=[100],
        reg=['F2', 'N3'],
        lr=[0.1],
        llr=[0.01, 0.05, 0.1],
        ct=['none', 'latent', 'graph'],
        mvl=[True],
        ls=[1],
        # i=['standard', 'reciprocal'],
        i=['standard'],
        V=[3],
        LE=[10],
        o=['adagrad'],
    )

    configurations = list(cartesian_product(hyp_space))

    path = 'logs/wn18rr/wn18rr_beaker_v3'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/pminervi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=48:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/meta-kbc

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 30 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

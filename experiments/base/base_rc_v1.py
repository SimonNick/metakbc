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
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python3 ./bin/kbc-cli.py ' \
        f'--train data/{c["data"]}/train.tsv ' \
        f'--dev data/{c["data"]}/dev.tsv ' \
        f'--test data/{c["data"]}/test.tsv ' \
        f'-m {c["m"]} -k {c["k"]} -b {c["b"]} -e {c["e"]} ' \
        f'--F2 {c["f2"]} --N3 {c["n3"]} -l {c["lr"]} -I {c["i"]} -V {c["V"]} -o {c["o"]} -q '
    return command


def to_logfile(c, path):
    outfile = "{}/rc_v1.{}.log".format(path, summary(c).replace("/", "_").replace(" ", "_"))
    return outfile


def main(argv):
    hyp_space_1 = dict(
        m=['complex'],
        data=['umls', 'nations', 'kinship'],
        k=[10, 50, 100, 500, 1000],
        b=[100],
        e=[100],
        f2=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        n3=[0],
        lr=[0.1],
        i=['standard', 'reciprocal'],
        V=[3],
        o=['adagrad'],
    )

    hyp_space_2 = dict(
        m=['complex'],
        data=['umls', 'nations', 'kinship'],
        k=[10, 50, 100, 500, 1000],
        b=[100],
        e=[100],
        f2=[0],
        n3=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
        lr=[0.1],
        i=['standard', 'reciprocal'],
        V=[3],
        o=['adagrad'],
    )

    configurations = list(cartesian_product(hyp_space_1)) + list(cartesian_product(hyp_space_2))

    path = '/home/ucacmin/Scratch/meta-kbc/logs/base/rc_v1'

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/ucacmin/'):
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    if os.path.exists('/Users/pasquale/'):
        path = './logs/base/rc_v1'

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            cmd = to_cmd(cfg)
            if cmd is not None:
                command_line = '{} > {} 2>&1'.format(cmd, logfile)
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
#$ -o /dev/null
#$ -e /dev/null
#$ -t 1-{}
#$ -l mem=8G
#$ -l h_rt=8:00:00

conda activate cpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/meta-kbc/

""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import signal
import subprocess

from ax.service.managed_loop import optimize

from typing import Dict, Optional, Tuple

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def handle_alarm(signum, frame):
    frame.f_locals['self'].kill()


def evaluate(p: Dict[str, float], f: Optional[float]) -> Dict[str, Tuple[float, Optional[float]]]:
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    mask_str = "--LM" if p["mvl"] else ""
    cmd_line = f'python3 ./bin/meta-cli.py ' \
        f'--train data/{p["data"]}/train.tsv --dev data/{p["data"]}/dev.tsv --test data/{p["data"]}/test.tsv ' \
        f'-m complex -k 1000 -b 100 -e 100 -R {p["reg"]} -l {p["lr"]} --LL {p["llr"]} ' \
        f'-W {p["feat"]} -S {p["s"]} {mask_str} --LE {p["le"]} -I standard -V 3 -o adagrad -q'

    max_time = 60 * 60
    # max_time = 15

    stdout = stderr = None
    signal.signal(signal.SIGALRM, handle_alarm)

    proc = subprocess.Popen(cmd_line, stdin=None, shell=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    signal.alarm(max_time)

    try:
        stdout, stderr = proc.communicate()
    except IOError:
        pass
    finally:
        signal.alarm(0)

    stdout_u = stdout.decode("utf-8")

    dev_test_pairs = []
    last_dev = None

    for line in stdout_u.split('\n'):
        tokens = line.split()

        if len(tokens) > 6:
            def search_for(s: str) -> Optional[float]:
                return float(tokens[5]) if tokens[2] == s else (float(tokens[4]) if tokens[1] == s else None)

            dev_value = search_for('dev')

            if dev_value is not None:
                last_dev = dev_value

            test_value = search_for('test')

            if test_value is not None:
                dev_test_pairs += [(last_dev, test_value)]
                last_dev = None

    best_dev = best_test = - np.inf

    for d, t in dev_test_pairs:
        if best_dev is None or d > best_dev:
            best_dev, best_test = d, t

    res = {'dev_MRR': (best_dev, 0.0), 'test_MRR': (best_test, 0.0)}
    return res


def main(argv):
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "data", "type": "fixed", "value": argv[0]},
            # {"name": "lr", "type": "range", "bounds": [1e-4, 1.0], "log_scale": True},
            {"name": "lr", "type": "fixed", "value": 0.1},
            {"name": "llr", "type": "range", "bounds": [1e-4, 1.0], "log_scale": True},
            {"name": "reg", "type": "choice", "values": ['F2', 'N3']},
            {"name": "feat", "type": "choice", "values": ['none', 'graph', 'latent']},
            # {"name": "s", "type": "choice", "values": [1, 2, 3]},
            {"name": "s", "type": "fixed", "value": 1},
            # {"name": "mvl", "type": "choice", "values": [True, False]},
            {"name": "mvl", "type": "fixed", "value": True},
            {"name": "le", "type": "fixed", "value": 10},
        ], evaluation_function=evaluate, objective_name='dev_MRR', total_trials=16)

    print(best_parameters)
    print(values)
    # print(experiment)
    # print(model)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])

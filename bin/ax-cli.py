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
    # If the alarm is triggered, we're still in the exec_proc.communicate()
    # call, so use exec_proc.kill() to end the process.
    frame.f_locals['self'].kill()


def evaluate(p: Dict[str, float], f: Optional[float]) -> Tuple[float, Optional[float]]:
    lr = p['lr']
    llr = p['llr']
    reg = p['reg']

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_line = f'python3 ./bin/meta-cli.py --train data/umls/dev.tsv --dev data/umls/dev.tsv --test data/umls/test.tsv ' \
        f'-m complex -k 1000 -b 100 -e 100 -R {reg} -l {lr} --LL {llr} -W graph -S 0 -I standard -V 1 -o adagrad -q'

    max_time = 60 * 10

    stdout = stderr = None
    signal.signal(signal.SIGALRM, handle_alarm)

    cmd_elements = cmd_line.split(' ')
    proc = subprocess.Popen(cmd_elements, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

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
            dev_value = None

            if tokens[2] == 'dev':
                dev_value = float(tokens[5])
            elif tokens[1] == 'dev':
                dev_value = float(tokens[4])

            if dev_value is not None:
                last_dev = dev_value

            test_value = None
            if tokens[2] == 'test':
                test_value = float(tokens[5])
            elif tokens[1] == 'test':
                test_value = float(tokens[4])

            if test_value is not None:
                dev_test_pairs += [(last_dev, test_value)]
                last_dev = None

    best_dev = best_test = - np.inf

    for d, t in dev_test_pairs:
        if best_dev is None or d > best_dev:
            best_dev = d
            best_test = t

    return best_test, None


def main(argv):
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "llr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
            {"name": "reg", "type": "choice", "values": ['F2', 'N3']},
        ], evaluation_function=evaluate, objective_name='MRR', total_trials=100)

    print(best_parameters)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])

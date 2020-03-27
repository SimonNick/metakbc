# -*- coding: utf-8 -*-

import os
import sys

import numpy as np

import subprocess
import pytest


@pytest.mark.light
def test_complex_cli_v1():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 ./bin/kbc-cli.py --train data/wn18rr/dev.tsv --dev data/wn18rr/dev.tsv ' \
              '--test data/wn18rr/test.tsv -m complex -k 100 -b 100 -e 1 --N3 0.0001 -l 0.1 -V 1 ' \
              '-o adagrad -B 2000'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    check = False

    for line in lines:
        if 'Batch 1/31' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 18.311768, atol=1e-3, rtol=1e-3)
        if 'Batch 10/31' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 18.273418, atol=1e-3, rtol=1e-3)
        if 'Final' in line and 'dev results' in line:
            value = float(line.split()[4])
            np.testing.assert_allclose(value, 0.116451, atol=1e-3, rtol=1e-3)

            check = True

    assert check


@pytest.mark.light
def test_complex_cli_v2():
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    cmd_str = 'python3 ./bin/kbc-cli.py --train data/umls/train.tsv --dev data/umls/dev.tsv ' \
              '--test data/umls/test.tsv -m complex -k 1000 -b 100 -e 5 --F2 0 --N3 0.005 ' \
              '-l 0.1 -I standard -V 1 -o adagrad -q'

    cmd = cmd_str.split()

    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()

    sys.stdout = sys.stderr

    lines = out.decode("utf-8").split("\n")

    check = False

    for line in lines:
        if 'Epoch 1/5' in line and 'Loss' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 8.3033, atol=1e-3, rtol=1e-3)
        if 'Epoch 1/5' in line and 'dev results' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 0.858015, atol=1e-3, rtol=1e-3)
        if 'Epoch 2/5' in line and 'Loss' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 6.3510, atol=1e-3, rtol=1e-3)
        if 'Epoch 2/5' in line and 'dev results' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 0.900180, atol=1e-3, rtol=1e-3)
        if 'Epoch 3/5' in line and 'Loss' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 6.0337, atol=1e-3, rtol=1e-3)
        if 'Epoch 3/5' in line and 'dev results' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 0.932358, atol=1e-3, rtol=1e-3)
        if 'Epoch 4/5' in line and 'Loss' in line:
            value = float(line.split()[3])
            np.testing.assert_allclose(value, 5.9025, atol=1e-3, rtol=1e-3)
        if 'Epoch 4/5' in line and 'dev results' in line:
            value = float(line.split()[5])
            np.testing.assert_allclose(value, 0.931345, atol=1e-3, rtol=1e-3)

            check = True

    assert check


if __name__ == '__main__':
    pytest.main([__file__])
    # test_complex_cli_v2()

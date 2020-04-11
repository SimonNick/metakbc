#!/usr/bin/env bash

rm -f umls.txt nations.txt kinship.txt

PYTHONPATH=. python3 ./bin/ax-cli.py umls > umls.txt 2>&1
PYTHONPATH=. python3 ./bin/ax-cli.py nations > nations.txt 2>&1
PYTHONPATH=. python3 ./bin/ax-cli.py kinship > kinship.txt 2>&1

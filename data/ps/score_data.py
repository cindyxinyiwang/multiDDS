import argparse
import os
import sys
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="display filtered data")
    parser.add_argument("score_file", type=Path)
    parser.add_argument("data_file", type=Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    score_file = args.score_file

    idx = []
    with open(score_file, 'r') as f1:
        score = f1.read()
    for _, item in enumerate(score.split('\n')):
        idx.append(int(item))

    with open(args.data_file, 'r') as f2:
        data = f2.read()

    # filter the src and target value by 20% manually
    # assert len(src_data.split('\n')) == len(trg_data.split('\n'))
    output = []
    for i, item in enumerate(data.split('\n')):
        if i in idx:
            output.append(item)

    print(output)

if __name__ == '__main__':
    main()
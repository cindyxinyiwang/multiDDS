import argparse
import os
import sys
from pathlib import Path
import random


def parse_args():
    parser = argparse.ArgumentParser(description="filter data manually")
    parser.add_argument("src_file", type=Path)
    parser.add_argument("trg_file", type=Path)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    src_file = args.src_file
    trg_file = args.trg_file

    with open(src_file, 'r') as f1:
        src_data = f1.read()
    with open(trg_file, 'r') as f2:
        trg_data = f2.read()

    # filter the src and target value by 20% manually
    # assert len(src_data.split('\n')) == len(trg_data.split('\n'))
    src_output = []
    trg_output = []
    for idx, item in enumerate(src_data.split('\n')):
        if idx % 10 == 0:
            src_output.append(item)
    for idx, item in enumerate(trg_data.split('\n')):
        if idx % 10 == 0:
            trg_output.append(item)

    # print(len(src_output))
    # # write into output file
    with open('zh_en.zh', 'w') as f:
        f.write("\n".join(src_output))
    with open('zh_en.en', 'w') as f:
        f.write("\n".join(trg_output))


if __name__ == '__main__':
    main()
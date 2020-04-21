import sentencepiece as spm
import os
import time
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_prefix", type=str)
parser.add_argument("--input", type=str)
parser.add_argument("--vocab_size", type=int)
parser.add_argument("--shuffle_input_sentence", type=str, default="false")
parser.add_argument("--input_sentence_size", type=int, default=-1)
args = parser.parse_args()

spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --shuffle_input_sentence={} --input_sentence_size={} --hard_vocab_limit=false'.format(args.input, args.model_prefix, args.vocab_size, args.shuffle_input_sentence,  args.input_sentence_size))

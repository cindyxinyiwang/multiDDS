import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument("--language-names", type=str, help="language codes in the available training data")
parser.add_argument("--file-pattern", type=str, help="file pattern of the training data; we automatically replace CODE with each language name")
parser.add_argument("--base-language-name", type=str, help="the name of the language we want to improve, typically the low resource language")
parser.add_argument("--filter-data", action='store_true', help="if set to true, also filter data based on the calculated distance")
parser.add_argument("--file-pattern-src", type=str, help="file pattern of the src training data; we automatically replace CODE with each language name")
parser.add_argument("--file-pattern-trg", type=str, help="file pattern of the trg training data; we automatically replace CODE with each language name")

args = parser.parse_args()

def get_vocab_set(language_name, file_pattern):
  file_name = file_pattern.replace("CODE", language_name)
  vocab = defaultdict(int)
  with open(file_name) as myfile:
    for line in myfile:
        toks = line.split()
        for tok in toks:
          for j in range(4):
            for i in range(len(tok)-j):
              ngram = tok[i:j]
              vocab[ngram] += 1
  sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])
  sorted_vocab = sorted_vocab[:min(10000, len(sorted_vocab))]
  v = set([item[0] for item in sorted_vocab])
  return v

if __name__ == "__main__":
  args.language_names = args.language_names.split(",")
  base_lan_vocab = get_vocab_set(args.base_language_name, args.file_pattern)

  dist = []
  for lan_name in args.language_names:
    lan_vocab = get_vocab_set(lan_name, args.file_pattern)
    x = lan_vocab.intersection(base_lan_vocab)
    print("{} {} {}".format(args.base_language_name, lan_name, len(x)))
    dist.append(len(x))

  if args.filter_data:
    data = {}
    for i,lan_name in enumerate(args.language_names):
      src_file = open(args.file_pattern_src.replace("CODE", lan_name), 'r')
      trg_file = open(args.file_pattern_trg.replace("CODE", lan_name), 'r')
      for src, trg in zip(src_file, trg_file):
        if trg not in data:
          data[trg] = [None for _ in range(len(args.language_names))]
        data[trg][i] = src
    for i,lan_name in enumerate(args.language_names):
      src_file = open(args.file_pattern_src.replace("CODE", lan_name) + ".filtered", 'w')
      trg_file = open(args.file_pattern_trg.replace("CODE", lan_name) + ".filtered", 'w')
      for trg in data.keys():
        srcs = data[trg]
        if srcs[i] is None:
          continue
        write = True
        for k, v in enumerate(srcs):
          if v is not None and dist[k] > dist[i]:
            write = False
            break
        if write:
          src_file.write(srcs[i])
          trg_file.write(trg)


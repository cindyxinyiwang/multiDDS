import numpy as np

#stats_file = "checkpoints/subsample/collect_stats/stats"
#src_file = "data/iwslt15_ende/train.teden"
#trg_file = "data/iwslt15_ende/train.tedde"
#sorted_src_file = "checkpoints/subsample/collect_stats/train.tedenavesort"
#sorted_trg_file = "checkpoints/subsample/collect_stats/train.teddeavesort"
#sorted_stats_file = "checkpoints/subsample/collect_stats/avesort"

stats_file = "checkpoints/continue_nmt/1o2/stats"
src_file = "data-bin/iwslt15_ende/train.tedenraw1o2-tedderaw1o2.tedenraw1o2"
trg_file = "data-bin/iwslt15_ende/train.tedenraw1o2-tedderaw1o2.tedderaw1o2"
sorted_src_file = "checkpoints/continue_nmt/1o2/train.teden1o2avesort"
sorted_trg_file = "checkpoints/continue_nmt/1o2/train.tedde1o2avesort"
sorted_stats_file = "checkpoints/continue_nmt/1o2/avesort"


stats = []
with open(stats_file, 'r') as myfile:
    for line in myfile:
        toks = line.split()
        if len(toks) > 1:
            toks = [float(t) for t in toks]
            stats.append(toks[10:])

stats = np.array(stats)
ave = np.mean(stats, axis=1)
sorted_idx = np.argsort(ave)

with open(sorted_stats_file, 'w') as myfile:
    for i in sorted_idx:
        myfile.write(str(ave[i]) + "\n")

src_data = open(src_file, 'r').readlines()
with open(sorted_src_file, 'w') as myfile:
    for i in sorted_idx:
        myfile.write(src_data[i].strip() + "\n")

trg_data = open(trg_file, 'r').readlines()
with open(sorted_trg_file, 'w') as myfile:
    for i in sorted_idx:
        myfile.write(trg_data[i].strip() + "\n")




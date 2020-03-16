import os
import operator

data_dir = "/projects/tir1/corpora/multiling-text/ted/"
lan_size = 9

data_count = {}
for d in os.listdir(data_dir):
  if d.endswith("eng") and len(d.split("_"))==2:
    lan = d.split("_")[0]
    if lan == "eng": continue
    train = data_dir + d + "/ted-train.orig.{}-eng".format(lan)
    num_lines = sum(1 for line in open(train))
    data_count[lan] = num_lines

interval = len(data_count) // lan_size
sorted_data_count = sorted(data_count.items(), key=operator.itemgetter(1))

for i in range(lan_size):
  print(sorted_data_count[min((i+1)*interval, len(sorted_data_count))])

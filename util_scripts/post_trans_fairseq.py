import sys

trans_file = sys.argv[1]

trans_file_ref = trans_file + ".ref"
trans_file_hyp = trans_file + ".hyp"

trans_file_ref = open(trans_file_ref, 'w')
trans_file_hyp = open(trans_file_hyp, 'w')

data_dict = {}
for line in open(trans_file, 'r'):
  if line.startswith("T-"):
    toks = line.split()
    idx = int(toks[0].split("-")[1])
    assert idx not in data_dict
    data_dict[idx] = [" ".join(toks[1:])]
  elif line.startswith("H-"):
    toks = line.split()
    idx = int(toks[0].split("-")[1])
    assert idx in data_dict
    assert len(data_dict[idx])==1
    data_dict[idx].append(" ".join(toks[2:]))

for i in range(len(data_dict)):
  trans_file_ref.write(data_dict[i][0] + '\n') 
  trans_file_hyp.write(data_dict[i][1] + '\n') 

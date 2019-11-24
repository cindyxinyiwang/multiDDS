import sys
import numpy as np

#nll_list = []
#words_list = []
#for line in sys.stdin:
#    if line.startswith("H-"):
#        toks = line.split()
#        nll = float(toks[1])
#        words_list.append(len(toks)-2)
#        nll_list.append(nll*(len(toks)-2) )
#
#
#print("num_lines={}".format(len(nll_list)))
#print("ppl={}".format(np.exp( sum(nll_list)/sum(words_list) )))
#

nll_list = []
ave_nll_list = []
num_lines = 0
for line in sys.stdin:
    if line.startswith("P-"):
        num_lines += 1
        toks = line.split()[1:]
        nlls = [ -float(t) for t in toks]
        nll_list.extend(nlls)
    if line.startswith("H-"):
        toks = line.split()
        ave_nll_list.append(-float(toks[1]))

print("num_lines={}".format( num_lines ))
print(sum(nll_list)/len(nll_list))
print("ppl={}".format(np.exp( sum(nll_list)/len(nll_list) )))
#print("ppl={}".format(pow(2, sum(nll_list)/len(nll_list) )))
#print("ppl={}".format(np.exp(sum(ave_nll_list)/len(ave_nll_list))))



import sys


filename = sys.argv[1]

words = {}
with open(filename, 'r') as myfile:
    for line in myfile:
        toks = line.split()
        for w in toks:
            if w in words: 
                words[w] += 1
            else:
                words[w] = 0

out_filename = filename + ".wordfreq"

sorted_words = sorted(words.items(), key=lambda kv:kv[1])[::-1]
with open(out_filename, 'w') as myfile:
    for w, count in sorted_words:
        myfile.write("{} {}\n".format(w, count))

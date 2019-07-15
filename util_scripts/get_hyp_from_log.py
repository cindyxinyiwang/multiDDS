import sys

prefix = sys.argv[1]
if prefix == "H-":
    for line in sys.stdin:
        if line.startswith(prefix):
            toks = line.split()
            print(" ".join(toks[2:]))
elif prefix == "T-":
    for line in sys.stdin:
        if line.startswith(prefix):
            toks = line.split()
            print(" ".join(toks[1:]))

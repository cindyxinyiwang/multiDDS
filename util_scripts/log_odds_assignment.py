import argparse

parser = argparse.ArgumentParser(description='Assign log odds to each sentence')
parser.add_argument('--log_odds', help='log odds file ', default='log_odds.txt')
parser.add_argument('--i', help='input file containing source lines', default='gradnorm.out')
parser.add_argument('--o', help='output', default='allreviewwords.out')
args = parser.parse_args()


input_file = open(args.i, 'r')
output_file = open(args.o, 'w')
w2logodds = {}
w = None
with open(args.log_odds, 'r') as myfile:
    for i, line in enumerate(myfile):
        if i%2 == 0: 
            w = line.strip()
        else:
            w2logodds[w] = float(line.strip())

words = []
for line in input_file:
    if line.startswith("S-"):
        output_file.write(line)
        words = line.split()[1:]
    elif line.startswith("T-"):
        output_file.write(line)
    elif line.startswith("N-"):
        logodds = [line.split()[0]]
        for w in words:
            if w in w2logodds:
                logodds.append(str(w2logodds[w]))
            else:
                logodds.append('0')
        output_file.write(" ".join(logodds) + "\n")


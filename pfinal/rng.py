import random
import sys

N = int(sys.argv[1])
F = sys.argv[2]

M = 2
B = 50
V = 10

with open(F, "w") as ofile:
    for _ in range(N):
        x = random.uniform(1, 100)
        # y = random.uniform(1, 100)
        y = M * x + B + random.gauss(0, V)
        ofile.write("%.2f\t%.2f\n" % (x, y))

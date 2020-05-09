import random
import sys

N = 150
days = 120
F = f'/u/wwang00/cs378h-concurrency/pfinal/input/random-{N}-{days}-input.txt'

prices = [random.uniform(100, 200) for _ in range(N)]

with open(F, "w") as ofile:
    ofile.write("%d\t%d\n" % (N, days))
    for _ in range(N):
        diff = [random.gauss(0, 1) for _ in range(N)]
        for i in range(len(prices)):
            prices[i] += diff[i]
            ofile.write("%.2f\t" % prices[i])
        ofile.write("\n")

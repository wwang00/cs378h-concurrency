import matplotlib.pyplot as plt

N = 10
avgs = {}
avgs_p = {}

# average samples
for p in range(100, 501, 100):
    with open(f"data/nb-{p}.dat", 'r') as ifile:
        avgs[p] = 0
        for line in ifile:
            avgs[p] += float(line)
        avgs[p] /= N
    with open(f"data/nb-{p}-p.dat", 'r') as ifile:
        avgs_p[p] = 0
        for line in ifile:
            avgs_p[p] += float(line)
        avgs_p[p] /= N

X = [x for x in range(100, 501, 100)]
Y = [avgs[p] / avgs_p[p] for p in range(100, 501, 100)]
plt.xlabel('points')
plt.ylabel('speedup')
plt.plot(X, Y, '-ro')
plt.show()

import matplotlib.pyplot as plt

N = 10
avgs = [0 for _ in range(17)]

# average samples
for n in range(1, 17):
    with open(f"data/nb-100-{n}.dat", 'r') as ifile:
        avgs[n] = 0
        for line in ifile:
            avgs[n] += float(line)
        avgs[n] /= N

X = [x for x in range(2, 17)]
Y = [avg / avgs[1] for avg in avgs[2:]]
plt.xlabel('procs')
plt.ylabel('speedup')
plt.plot(X, Y, '-ro')
plt.show()

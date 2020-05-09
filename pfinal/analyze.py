import matplotlib.pyplot as plt

N = 20.0
X = [20, 50, 100, 150, 200]
seq = {}
cuda = {}

# average samples
for p in X: 
    with open(f"data/seq-{p}.dat", 'r') as ifile:
        seq[p] = 0
        for line in ifile:
            seq[p] += float(line)
        seq[p] /= N
    with open(f"data/cuda-{p}.dat", 'r') as ifile:
        cuda[p] = 0
        for line in ifile:
            cuda[p] += float(line)
        cuda[p] /= N

Y = [seq[p] / cuda[p] for p in X]
plt.xlabel('symbols')
plt.ylabel('speedup')
plt.plot(X, Y, '-ro')
#plt.show()

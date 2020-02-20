import matplotlib.pyplot as plt

N = 20
X = [2, 4, 8, 16]
X1 = [1] + X
TYPES = {"simple", "coarse", "fine"}

hash_times = {typ: {} for typ in TYPES}
group_times = {typ: {} for typ in TYPES}
comp_times = {typ: {} for typ in TYPES}

for typ in TYPES:
    for x in X1:
        with open(f"output/{typ}_hash_{x}.txt", "r") as fin:
            hash_times[typ][x] = sum(int(t) for t in fin) / N
        with open(f"output/{typ}_comp_{x}.txt", "r") as fin:
            comp_times[typ][x] = sum(int(t) for t in fin) / N

plt.xlabel('threads')
plt.ylabel('speedup')

plt.plot(X, [(group_times["simple"][1] / group_times["simple"][x])
             for x in X], '-ro', label=f"group_simple")
plt.plot(X, [(group_times["coarse"][1] / group_times["coarse"][x])
             for x in X], '-g^', label=f"group_coarse")
plt.plot(X, [(group_times["fine"][1] / group_times["fine"][x])
             for x in X], '-bs', label=f"group_fine")
plt.legend(loc="upper left")
plt.show()

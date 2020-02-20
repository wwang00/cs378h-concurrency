import matplotlib.pyplot as plt

N = 20
X = [1, 2, 4, 8, 16]
X0 = X[0:len(X) - 1]
X1 = X[1:len(X)]
TYPES = ['simple', 'coarse', 'fine']
FMTS = ['-ro', '-g^', '-bs']

hash_times = {typ: {} for typ in TYPES}
comp_times = {typ: {} for typ in TYPES}
group_mutex_times = {typ: {} for typ in TYPES}
group_chan_times = {typ: {} for typ in TYPES}
group_chan_k_times = {typ: {} for typ in TYPES}

for typ in TYPES:
    for x in X:
        with open(f"output/{typ}_hash_{x}.txt", "r") as fin:
            hash_times[typ][x] = sum(int(t) for t in fin) / N
        with open(f"output/{typ}_comp_{x}.txt", "r") as fin:
            comp_times[typ][x] = sum(int(t) for t in fin) / N
        with open(f"output/{typ}_group_{x}_{x}.txt", "r") as fin:
            group_mutex_times[typ][x] = sum(int(t) for t in fin) / N
        with open(f"output/{typ}_group_{x}_1.txt", "r") as fin:
            group_chan_times[typ][x] = sum(int(t) for t in fin) / N
        with open(f"output/{typ}_group_16_{x}.txt", "r") as fin:
            group_chan_k_times[typ][x] = sum(int(t) for t in fin) / N

hash_seq_times = {typ: hash_times[typ][1] for typ in TYPES}

plt.xlabel('hash_workers')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X1, [(hash_seq_times[typ] / hash_times[typ][x])
                  for x in X1], fmt, label=f"hash_{typ}")
plt.legend(loc='upper left')
plt.savefig('data/hash.png', format='png', dpi=255)
plt.clf()

comp_seq_times = {typ: comp_times[typ][1] for typ in TYPES}

plt.xlabel('comp_workers')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X1, [(comp_seq_times[typ] / comp_times[typ][x])
                  for x in X1], fmt, label=f"comp_{typ}")
plt.legend(loc='center right')
plt.savefig('data/comp.png', format='png', dpi=255)
plt.clf()

group_seq_times = {typ: group_mutex_times[typ][1] for typ in TYPES}

plt.xlabel('workers')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X1, [(group_seq_times[typ] / group_mutex_times[typ][x])
                  for x in X1], fmt, label=f"group_mutex_{typ}")
plt.legend(loc='center right')
plt.savefig('data/group_mutex.png', format='png', dpi=255)
plt.clf()

plt.xlabel('hash_workers')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X1, [(group_seq_times[typ] / group_chan_times[typ][x])
                  for x in X1], fmt, label=f"group_chan_{typ}")
plt.legend(loc='center right')
plt.savefig('data/group_chan.png', format='png', dpi=255)
plt.clf()

plt.xlabel('data_workers')
plt.ylabel('speedup')
for (typ, fmt) in zip(TYPES, FMTS):
    plt.plot(X0, [(group_seq_times[typ] / group_chan_k_times[typ][x])
                  for x in X0], fmt, label=f"group_chan_k_{typ}")
plt.legend(loc='center right')
plt.savefig('data/group_chan_k.png', format='png', dpi=255)
plt.clf()

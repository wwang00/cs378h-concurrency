import matplotlib.pyplot as plt

int_10000_s = []
int_10000 = []

with open("10000_ints_we_s.avg", "r") as fin:
    int_10000_s = [int(x) for x in fin]

with open("10000_ints_we.avg", "r") as fin:
    int_10000 = [int(x) for x in fin]

x = range(0, 34, 2)

plt.xlabel('threads')
plt.ylabel('microseconds')

plt.plot(x, int_10000_s, '-ro', label="10000_ints_custom")
plt.plot(x, int_10000, '-g^', label="10000_ints_pthread")
plt.legend(loc="upper left")
plt.savefig()

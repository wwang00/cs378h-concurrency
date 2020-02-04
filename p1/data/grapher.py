import matplotlib.pyplot as plt

ten = []
hundred = []
thousand = []

with open("10_10_floats_we.avg", "r") as fin:
    ten = [int(x) for x in fin]

with open("100_100_floats_we.avg", "r") as fin:
    hundred = [int(x) for x in fin]

with open("1000_1000_floats_we.avg", "r") as fin:
    thousand = [int(x) for x in fin]

x = range(0, 34, 2)

plt.plot(x, ten, '-ro', x, hundred, '-g^', x, thousand, '-bs')
plt.xlabel('threads')
plt.ylabel('microseconds')
plt.show()

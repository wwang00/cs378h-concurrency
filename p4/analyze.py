import matplotlib.pyplot as plt

N = 10
X = [1, 2, 4, 6, 8, 10, 12]
times = [1413, 903, 2667, 4113, 5236, 6994, 7859]
props = [85, 90, 74, 63, 47, 43, 32]

plt.xlabel('participants')
plt.ylabel('commit_percentage')
plt.plot(X, [props[x] for x in range(0, 7)], '-ro')
# plt.legend(loc='center right')
plt.show()
#plt.savefig('data/plot.png', format='png', dpi=255)
# plt.clf()

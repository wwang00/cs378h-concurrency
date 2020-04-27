import matplotlib.pyplot as plt

F = "input/input.txt"

pts = []

with open(F, "r") as ifile:
    ifile.readline()
    for _ in range(36):
        pt = ifile.readline().split(",")
        pts.append((pt[0], pt[1]))
pts.sort(key=lambda p : p[0])

plt.plot([p[0] for p in pts], [p[1] for p in pts])
plt.show()
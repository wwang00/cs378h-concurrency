from random import uniform

N = 1000
M = 2
B = 50
V = 20

F = "input/input.txt"

with open(F, "w") as ofile:
    ofile.write(f"{N}\n")
    for _ in range(N):
        x = uniform(1, 100)
        y = M * x + B + uniform(-V, V)
        ofile.write("%.2f,%.2f\n" % (x, y))

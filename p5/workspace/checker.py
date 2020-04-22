from math import sqrt

X = 0.000031
Y = 0.526151
M = 17.556
G = 0.0001

x, y = 2.747091, 2.355906
m = 19.304

dx = x - X
dy = y - Y
d = sqrt(dx ** 2 + dy ** 2)
c = G * M * m / (d ** 3)
print(c * dx)
print(c * dy)

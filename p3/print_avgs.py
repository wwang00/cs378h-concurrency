import sys

with open(sys.argv[1]) as f:
    total = 0
    count = 0
    for t in f:
        total += int(t)
        count += 1
    print(int(total / count))

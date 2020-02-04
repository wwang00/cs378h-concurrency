total = 0

with open("10000_10_floats_we.avg", "w") as fout:
    for i in range(0, 34, 2):
        with open("10000_10_floats_" + str(i) + "_we.data", "r") as fin:
            total = sum(int(x) for x in fin)

        avg = int(total / 100)
        print(avg)
        fout.write(str(avg) + "\n")

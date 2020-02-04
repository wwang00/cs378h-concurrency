total = 0

with open("10000_ints_we_s.avg", "w") as fout:
    for i in range(0, 34, 2):
        with open("10000_ints_" + str(i) + "_we_s.data", "r") as fin:
            total = sum(int(x) for x in fin)

        avg = int(total / 2)
        print(avg)
        fout.write(str(avg) + "\n")

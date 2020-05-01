IF='/Users/william/Downloads/EWC.csv'
OF='/Users/william/cs378h-concurrency/pfinal/input/EWC.txt'

with open(IF, 'r') as ifile:
    with open(OF, 'w') as ofile:
        first = True
        for line in ifile:
            if first:
                first = False
                continue
            data = line.split(',')
            close = data[4]
            ofile.write(f"{close}\n")
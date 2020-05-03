SYMB = 'EWC'

IF = f"/Users/william/Downloads/{SYMB}-2.csv"
OF = f"/Users/william/cs378h-concurrency/pfinal/input/{SYMB}-input.txt"

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

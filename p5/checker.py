F = 'data.dat'

avgs = {}

with open(f"data/{F}", 'r') as ifile:
    last_key = None
    for line in ifile:
        if last_key is None:
            last_key = int(line)
        else:
            if last_key in avgs.keys():
                avgs[last_key] += float(line)
            else:
                avgs[last_key] = float(line)
            last_key = None

print(avgs)
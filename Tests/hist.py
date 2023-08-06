import sys
import json

import numpy as np
import matplotlib.pyplot as plt

with open(sys.argv[1]) as f:
    data = json.load(f)

toHist = sys.argv[2]

if toHist == 'nc':
    toHist = "nativeContacts"
elif toHist == 'bonds':
    toHist = "bonds"
else:
    print("Error: invalid argument")
    exit()

typ = data["forceField"][toHist]["data"]
typ = np.array(typ)

dst = typ[:, 2]
n   = typ[:, 3]

plt.hist(dst, bins=50)
plt.show()

plt.hist(n, bins=50)
plt.show()

plt.plot(dst, n, "o")
plt.show()

#Count the number of different n and plot x-n , y-number of n
ntypes = np.unique(n)
ncount = np.zeros(len(ntypes))
for i in range(len(ntypes)):
    ncount[i] = np.sum(n == ntypes[i])

plt.plot(ntypes, ncount, "o")

for i in range(len(ntypes)):
    print(ntypes[i], ncount[i])

plt.show()


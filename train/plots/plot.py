import json
import os
import matplotlib.pyplot as plt

data = {}
margin, r, rho, tau = 0, 0, 0, 0
with open("eval_prob.txt", "r") as f:
    for line in f.readlines():
        if "Margin = " in line:
            margin = float(line.strip()[9:])
        if "Pearson" in line:
            r = float(line.strip().split('statistic=')[1].split(', ')[0])
        if "Spearman" in line:
            rho = float(line.strip().split('statistic=')[1].split(', ')[0])
        if "Kendall" in line:
            tau = float(line.strip().split('statistic=')[1].split(', ')[0])
            data[margin] = (r, rho, tau)
    
# print(data.keys())
plt.plot(list(data.keys()), [v[0] for v in data.values()], label="Pearson\'s r", marker='.', markersize=3)
plt.plot(list(data.keys()), [v[1] for v in data.values()], label="Spearman\'s rho", marker='.', markersize=3)
plt.plot(list(data.keys()), [v[2] for v in data.values()], label="Kendall\'s tau", marker='.', markersize=3)
plt.legend()
plt.xlabel("margin")
plt.ylabel("statistic")
plt.savefig("plot.png", dpi=300)
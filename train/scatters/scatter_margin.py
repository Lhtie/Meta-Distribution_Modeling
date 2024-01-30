import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

scorer = [
    "margin=0-epoch-0",
    "margin=0.4-epoch-0",
    "margin=0.6-epoch-0",
    "margin=0.8-epoch-0",
    "margin=1.0-epoch-0",
    "margin=1.6-epoch-0",
    "margin=2.4-epoch-0",
    "margin=4.2-epoch-0",
    "margin=6.4-epoch-0",
    "margin=8.4-epoch-0",
    "margin=10.2-epoch-0",
    "margin=12.8-epoch-0",
]
eval_model = [
    "commongen_pythia-160m"
] * 12

fig = plt.figure(figsize=(16, 12))
axes = fig.subplots(3, 4)
flattened_axes = [y for x in axes for y in x]
for ax, s, t in zip(flattened_axes, scorer, eval_model):
    with open(f"../.cache/sent_scores-{s}-{t}.pt", "rb") as f:
        scores = pickle.load(f)
        
    ax.scatter(scores[0], scores[1], s=8, label='-'.join(s.split("-")[:1]))
    ax.set_xlabel("Scorer")
    ax.set_ylabel("BLEU")
    ax.legend()
    
plt.tight_layout()
plt.savefig(f"corr_prob_margin.png", dpi=300)
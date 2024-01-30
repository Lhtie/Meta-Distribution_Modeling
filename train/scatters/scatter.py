import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

scorer = [
    # "models-gpt2",
    "scorer_prob-epoch-0",
    "scorer_prob-epoch-1",
    # "scorer_prob-epoch-2",
    # "scorer_prob-epoch-3",
    # "scorer_prob-epoch-4"
]
eval_model = [
    "commongen_pythia-160m"
] * 2

for s, t in zip(scorer, eval_model):
    with open(f"../.cache/sent_scores-{s}-{t}.pt", "rb") as f:
        scores = pickle.load(f)
        
    plt.scatter(scores[0], scores[1], s=8, label='-'.join(s.split("-")[-2:]))
    plt.xlabel("Scorer")
    plt.ylabel("BLEU")
    
plt.legend()
    
plt.tight_layout()
plt.savefig(f"corr_prob.png", dpi=300)
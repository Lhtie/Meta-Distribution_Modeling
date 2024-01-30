import os
import json

for file in ["commongen.train.jsonl", "commongen.dev.jsonl"]:
    output_file = file.split('.')[1]
    ipts, opts = [], []
    with open(file, "r") as f:
        for line in f.readlines():
            d = json.loads(line)
            for scene in d['scene']:
                ipts.append(d['concept_set'].replace('#', ' '))
                opts.append(scene)
                
    with open(output_file + "_ipt.txt", "w") as f:
        f.writelines('\n'.join(ipts))
    with open(output_file + "_opt.txt", "w") as f:
        f.writelines('\n'.join(opts))
        
ipts, opts = [], []
with open("commongen.test_noref.jsonl", "r") as f:
    for line in f.readlines():
        d = json.loads(line.replace(',', ''))
        ipts.append(d['concept_set'].replace('#', ' '))
        
with open("test_ipt.txt", "w") as f:
    f.writelines('\n'.join(ipts))
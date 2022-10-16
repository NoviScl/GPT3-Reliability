import os
import json
from tqdm import tqdm

task_dir = "../NatInst/tasks/"
tasks = os.listdir(task_dir)
reasoning = {}

for task in tqdm(tasks):
    taskfile = os.path.join(task_dir, task)
    if taskfile[-4 : ] == "json":
        with open(taskfile, "r") as f:
            data = json.load(f)
        
        if "natural" in taskfile.lower():
            print (taskfile)
        
        reason = data["Reasoning"]
        if len(reason) < 1:
            continue 

        # if len(reason) > 1:
        #     print (reason, task)

        if reason[0] not in reasoning:
            reasoning[reason[0]] = [task]
        else:
            reasoning[reason[0]].append(task)
        
# for k,v in reasoning.items():
#     print (k,v)

# for k in reasoning.keys():
#     # if "multihop" in k.lower():
#     #     print (reasoning[k])
#     if "num" in k.lower():
#         print (k, reasoning[k])
    

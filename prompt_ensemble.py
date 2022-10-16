import os
import json
import random
from utils import *
random.seed(2022)

from mmlu_categories import *
maincat = list(categories.keys())
subcat = subcategories.values()
subcat = [eg[0] for eg in subcat]

def parse_log(logfile):
    all_subsets = {}
    with open(logfile, "r") as f:
        lines = f.readlines()
    
    subset = None
    for i, line in enumerate(lines):
        if "Current Subset:" in line:
            subset = line.split()[-1].strip()
            all_subsets[subset] = []
        if subset:
            if "Gold answer:" in line:
                eg = {}
                pred = lines[i-2].split()[-1].strip()
                eg["pred"] = pred 
                gold = line.split()[-1].strip()
                eg["gold"] = gold 
                token_probs = []
                for j in range(4):
                    token_probs.append(float(lines[i+2+j].split()[-1].replace(',', '')))
                eg["token_probs"] = token_probs 
                final_prob = float(lines[i+7].split()[-1].strip())
                eg["final_prob"] = final_prob 
                em = float(lines[i+8].split()[-1].strip())
                eg["em"] = em 
            
                all_subsets[subset].append(eg)
    
    return all_subsets

def oracle_ensemble(set_a, set_b):
    subcat_ems = {}
    maincat_ems = {}
    for k,v in set_a.items():
        ems = []
        v_a = v 
        v_b = set_b[k]
        for i_eg in range(len(v_a)):
            ems.append(max(v_a[i_eg]["em"], v_b[i_eg]["em"]))
        print (k, np.mean(ems))
        
        subcat_ = subcategories[k][0]
        maincat_ = None
        for k_,v_ in categories.items():
            if subcat_ in v_:
                maincat_ = k_

        if subcat_ not in subcat_ems:
            subcat_ems[subcat_] = [np.mean(ems)]
        else:
            subcat_ems[subcat_].append(np.mean(ems))
        
        if maincat_ not in maincat_ems:
            maincat_ems[maincat_] = [np.mean(ems)]
        else:
            maincat_ems[maincat_].append(np.mean(ems))
    
    for k,v in subcat_ems.items():
        print (k, np.mean(v))
    
    print ("\n")
    means = []
    for k,v in maincat_ems.items():
        print (k, np.mean(v))
        means.append(np.mean(v))
    print ("AVG: ", np.mean(means))
    
    
def max_ensemble(set_a, set_b):
    subcat_ems = {}
    maincat_ems = {}
    for k,v in set_a.items():
        ems = []
        v_a = v 
        v_b = set_b[k]
        for i_eg in range(len(v_a)):
            if v_a[i_eg]["final_prob"] >= v_b[i_eg]["final_prob"]:
                ems.append(v_a[i_eg]["em"])
            else:
                ems.append(v_b[i_eg]["em"])
        print (k, np.mean(ems))
        
        subcat_ = subcategories[k][0]
        maincat_ = None
        for k_,v_ in categories.items():
            if subcat_ in v_:
                maincat_ = k_

        if subcat_ not in subcat_ems:
            subcat_ems[subcat_] = [np.mean(ems)]
        else:
            subcat_ems[subcat_].append(np.mean(ems))
        
        if maincat_ not in maincat_ems:
            maincat_ems[maincat_] = [np.mean(ems)]
        else:
            maincat_ems[maincat_].append(np.mean(ems))
    
    for k,v in subcat_ems.items():
        print (k, np.mean(v))
    
    print ("\n")
    means = []
    for k,v in maincat_ems.items():
        print (k, np.mean(v))
        means.append(np.mean(v))
    print ("AVG: ", np.mean(means))
    

    
def interpolate_ensemble(set_a, set_b):
    subcat_ems = {}
    maincat_ems = {}
    for k,v in set_a.items():
        ems = []
        v_a = v 
        v_b = set_b[k]
        for i_eg in range(len(v_a)):
            probs = [np.mean([v_a[i_eg]["token_probs"][j], v_b[i_eg]["token_probs"][j]]) for j in range(4)]
            chosen = np.argmax(probs)
            # print (probs)
            # print (chosen)
            # print ()
            chosen = ["A", "B", "C", "D"][chosen]
            ems.append(chosen == v_a[i_eg]["gold"])
        print (k, np.mean(ems))
        
        subcat_ = subcategories[k][0]
        maincat_ = None
        for k_,v_ in categories.items():
            if subcat_ in v_:
                maincat_ = k_

        if subcat_ not in subcat_ems:
            subcat_ems[subcat_] = [np.mean(ems)]
        else:
            subcat_ems[subcat_].append(np.mean(ems))
        
        if maincat_ not in maincat_ems:
            maincat_ems[maincat_] = [np.mean(ems)]
        else:
            maincat_ems[maincat_].append(np.mean(ems))
    
    for k,v in subcat_ems.items():
        print (k, np.mean(v))
    
    print ("\n")
    means = []
    for k,v in maincat_ems.items():
        print (k, np.mean(v))
        means.append(np.mean(v))
    print ("AVG: ", np.mean(means))
    


all_subsets_closed_book = parse_log("/home/sichenglei/PromptQA/logs/MMLU/mmlu_all_closed_book_code002_5shot.log")
all_subsets_retrieval = parse_log("/home/sichenglei/PromptQA/logs/MMLU/mmlu_all_contriever_prompt_contriever_5shot.log")
interpolate_ensemble(all_subsets_closed_book, all_subsets_retrieval)



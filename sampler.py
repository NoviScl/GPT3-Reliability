import os
import sys
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
import re
import random
from tqdm import tqdm
from utils import *
from os import listdir
from os.path import isfile, join
csv.field_size_limit(sys.maxsize)
import string
puncs = list(string.punctuation) + [',', "'", '!', '.', '%', '\n', ' ']

from mmlu_categories import * 

random.seed(2022)

train_mappings = {
    "nq": "/data3/private/clsi/qa_data/nqopen/train.json",
    "triviaqa": "/data3/private/clsi/qa_data/triviaqa/train.json",
    "squad": "/data3/private/clsi/qa_data/squad_open/train.json",
    "hotpotqa": "/data3/private/clsi/hotpotqa/hotpot_train_v1.1.json",
    "webq": "/home/sichenglei/QAdatasets/WebQ/webq_train.json",
    "cwq": "/home/sichenglei/QAdatasets/ComplexWebQuestions_train.json",
    "qampari": "/home/sichenglei/QAdatasets/qampari_data/train_data.jsonl",
    "ambigqa": "/home/sichenglei/QAdatasets/train_light.json",
    "boolq": "/home/sichenglei/QAdatasets/BoolQ/train.jsonl",
    "webqsp": "/home/sichenglei/QAdatasets/WebQSP/data/WebQSP.train.json",
    "timeqa-easy": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/train.easy.json",
    "timeqa-hard": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/train.hard.json",
    "timeqa-human-easy": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_train.easy.json",
    "timeqa-human-hard": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_train.hard.json",
    "grailqa": "/home/sichenglei/QAdatasets/GrailQA_v1.0/grailqa_v1.0_train.json",
    "cfq-mcd1": "/home/sichenglei/QAdatasets/cfq/splits/mcd1_train.json",
    "cfq-mcd2": "/home/sichenglei/QAdatasets/cfq/splits/mcd2_train.json",
    "cfq-mcd3": "/home/sichenglei/QAdatasets/cfq/splits/mcd3_train.json",
    "freebaseqa": "/home/sichenglei/QAdatasets/FreebaseQA/FreebaseQA-train.json",
    "hybridqa": "/home/sichenglei/QAdatasets2/HybridQA/released_data/train.json",
    "ambigqa": "/home/sichenglei/QAdatasets/train_light.json",
    "reclor": "/home/sichenglei/QAdatasets/reclor_data/train.json",
    "race": ["/home/sichenglei/QAdatasets/RACE/train/high", "/home/sichenglei/QAdatasets/RACE/train/middle"],
    "boolq-rc": "/data3/private/clsi/UnifiedQAdata/boolq/train.tsv",
    "nq-dpr": "/data3/private/clsi/UnifiedQAdata/natural_questions_with_dpr_para/train.tsv",
    "subqa-overall": "/home/sichenglei/QAdatasets/subqa-data/dev_ori.json",
    "subqa-sub1": "/home/sichenglei/QAdatasets/subqa-data/dev_sub1.json",
    "subqa-sub2": "/home/sichenglei/QAdatasets/subqa-data/dev_sub2.json",
    "subqa-overall-passage": "/home/sichenglei/QAdatasets/subqa-data/dev_ori.json",
    "subqa-all10": "/home/sichenglei/QAdatasets/subqa-data/dev_ori.json",
    "mrqa-nq-train": "/home/sichenglei/MRQA/substituteEntity/src/datasets/normalized/MRQANaturalQuestionsTrain.jsonl",
    "mrqa-squad-train": "/home/sichenglei/MRQA/substituteEntity/src/datasets/normalized/MRQASQuADTrain.jsonl",
    "mrqa-newsqa-train": "/home/sichenglei/MRQA/substituteEntity/src/datasets/normalized/MRQANewsQATrain.jsonl",
}

test_mappings = {
    "nq": "/data3/private/clsi/qa_data/nqopen/dev.json",
    "triviaqa": "/data3/private/clsi/qa_data/triviaqa_alias/test.json",
    "squad": "/data3/private/clsi/qa_data/squad_open/test.json",
    "hotpotqa": "/data3/private/clsi/hotpotqa/hotpot_dev_fullwiki_v1.json",
    "webq": "/home/sichenglei/QAdatasets/WebQ/webq_test.json",
    "cwq": "/home/sichenglei/QAdatasets/ComplexWebQuestions_dev.json",
    "qampari": "/home/sichenglei/QAdatasets/qampari_data/test_data.jsonl",
    "ambigqa": "/home/sichenglei/QAdatasets/dev_light.json",
    "boolq": "/home/sichenglei/QAdatasets/BoolQ/dev.jsonl",
    "webqsp": "/home/sichenglei/QAdatasets/WebQSP/data/WebQSP.test.json",
    "timeqa-easy": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/test.easy.json",
    "timeqa-hard": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/test.hard.json",
    "timeqa-human-easy": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_test.easy.json",
    "timeqa-human-hard": "/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_test.hard.json",
    "grailqa": "/home/sichenglei/QAdatasets/GrailQA_v1.0/grailqa_v1.0_dev.json",
    "cfq-mcd1": "/home/sichenglei/QAdatasets/cfq/splits/mcd1_test.json",
    "cfq-mcd2": "/home/sichenglei/QAdatasets/cfq/splits/mcd2_test.json",
    "cfq-mcd3": "/home/sichenglei/QAdatasets/cfq/splits/mcd3_test.json",
    "freebaseqa": "/home/sichenglei/QAdatasets/FreebaseQA/FreebaseQA-eval.json",
    "hybridqa": "/home/sichenglei/QAdatasets2/HybridQA/released_data/dev.json",
    "ambigqa": "/home/sichenglei/QAdatasets/dev_light.json",
    "reclor": "/home/sichenglei/QAdatasets/reclor_data/val.json",
    "race": ["/home/sichenglei/QAdatasets/RACE/test/high", "/home/sichenglei/QAdatasets/RACE/test/middle"],
    "boolq-rc": "/data3/private/clsi/UnifiedQAdata/boolq/dev.tsv",
    "nq-dpr": "/data3/private/clsi/UnifiedQAdata/natural_questions_with_dpr_para/test.tsv",
}

subset_mappings = {
    "nq": "/home/sichenglei/PromptQA/testsets/nq.json",
    "triviaqa": "/home/sichenglei/PromptQA/testsets/triviaqa.json",
    "squad": "/home/sichenglei/PromptQA/testsets/squad.json",
    "hotpotqa": "/home/sichenglei/PromptQA/testsets/hotpotqa.json",
    "webq": "/home/sichenglei/PromptQA/testsets/webq.json",
    "cwq": "/home/sichenglei/PromptQA/testsets/cwq.json",
    "qampari": "/home/sichenglei/PromptQA/testsets/qampari.json",
    "boolq": "/home/sichenglei/PromptQA/testsets/boolq.json",
    "webqsp": "/home/sichenglei/PromptQA/testsets/webqsp.json",
    "timeqa-easy": "/home/sichenglei/PromptQA/testsets/timeqa/easy_test.json",
    "timeqa-hard": "/home/sichenglei/PromptQA/testsets/timeqa/hard_test.json",
    "timeqa-human-easy": "/home/sichenglei/PromptQA/testsets/timeqa/human_easy_test.json",
    "timeqa-human-hard": "/home/sichenglei/PromptQA/testsets/timeqa/human_hard_test.json",
    "grailqa": "/home/sichenglei/PromptQA/testsets/grailqa.json",
    "cfq-mcd1": "/home/sichenglei/PromptQA/testsets/cfq-mcd1.json",
    "cfq-mcd2": "/home/sichenglei/PromptQA/testsets/cfq-mcd2.json",
    "cfq-mcd3": "/home/sichenglei/PromptQA/testsets/cfq-mcd3.json",
    "freebaseqa": "/home/sichenglei/PromptQA/testsets/freebaseqa.json",
    "hybridqa": "/home/sichenglei/PromptQA/testsets/hybridqa.json",
    "ambigqa": "/home/sichenglei/PromptQA/testsets/ambigqa.json",
    "reclor": "/home/sichenglei/PromptQA/testsets/reclor.json",
    "race": "/home/sichenglei/PromptQA/testsets/race.json",
    "boolq-rc": "/home/sichenglei/PromptQA/testsets/boolq-rc.json",
    "nq-dpr": "/home/sichenglei/PromptQA/testsets/nq-dpr.json",
    "subqa-overall": "/home/sichenglei/PromptQA/testsets/subqa-overall.json",
    "subqa-sub1": "/home/sichenglei/PromptQA/testsets/subqa-sub1.json",
    "subqa-sub2": "/home/sichenglei/PromptQA/testsets/subqa-sub2.json",
    "subqa-overall-passage": "/home/sichenglei/PromptQA/testsets/subqa-overall-passage.json",
    "subqa-all10": "/home/sichenglei/PromptQA/testsets/subqa-all10.json",
    "mrqa-nq-train": "/home/sichenglei/PromptQA/testsets/mrqa-nq-train.json",
    "mrqa-squad-train": "/home/sichenglei/PromptQA/testsets/mrqa-squad-train.json",
    "mrqa-newsqa-train": "/home/sichenglei/PromptQA/testsets/mrqa-newsqa-train.json",
}

train_for_inference_mappings = {
    "nq": "/home/sichenglei/PromptQA/testsets/nq_dev.json",
    "hotpotqa": "/home/sichenglei/PromptQA/testsets/hotpotqa_train.json",
    "webqsp": "/home/sichenglei/PromptQA/testsets/webqsp_train.json",
}

glue_train_mappings = {
    "mnli": "/home/sichenglei/AdvGLUE/GLUE/MNLI/train.tsv",
    "mnli-mm": "/home/sichenglei/AdvGLUE/GLUE/MNLI/train.tsv",
    "qnli": "/home/sichenglei/AdvGLUE/GLUE/QNLI/train.tsv",
    "qqp": "/home/sichenglei/AdvGLUE/GLUE/QQP/train.tsv",
    "rte": "/home/sichenglei/AdvGLUE/GLUE/RTE/train.tsv",
    "sst2": "/home/sichenglei/AdvGLUE/GLUE/SST-2/train.tsv",
    "mrpc": "/home/sichenglei/LM-BFF/data/original/MRPC/train.tsv",
    "snli": "/home/sichenglei/LM-BFF/data/original/SNLI/train.tsv",
    "rte": "/home/sichenglei/LM-BFF/data/original/RTE/train.tsv",
    "scitail": "/home/sichenglei/MoreNLI/SciTail/train.tsv",
    "qnli": "/home/sichenglei/MoreNLI/QNLI/train.tsv",
    "wnli": "/home/sichenglei/MoreNLI/WNLI/train.tsv",
}

glue_test_mappings = {
    "mnli":  "/home/sichenglei/AdvGLUE/GLUE/MNLI/dev_matched.tsv",
    "mnli-mm": "/home/sichenglei/AdvGLUE/GLUE/MNLI/dev_mismatched.tsv",
    "qnli": "/home/sichenglei/AdvGLUE/GLUE/QNLI/dev.tsv",
    "qqp": "/home/sichenglei/AdvGLUE/GLUE/QQP/dev.tsv",
    "rte": "/home/sichenglei/AdvGLUE/GLUE/RTE/dev.tsv",
    "sst2": "/home/sichenglei/AdvGLUE/GLUE/SST-2/dev.tsv",
    "mrpc": "/home/sichenglei/LM-BFF/data/original/MRPC/dev.tsv",
    "snli": "/home/sichenglei/LM-BFF/data/original/SNLI/test.tsv",
    "rte": "/home/sichenglei/LM-BFF/data/original/RTE/dev.tsv",
    "scitail": "/home/sichenglei/MoreNLI/SciTail/dev.tsv",
    "qnli": "/home/sichenglei/MoreNLI/QNLI/dev.tsv",
    "wnli": "/home/sichenglei/MoreNLI/WNLI/dev.tsv",
}

glue_subset_mappings = {
    "mnli": "/home/sichenglei/PromptQA/testsets/GLUE/mnli_matched.json",
    "mnli-mm":  "/home/sichenglei/PromptQA/testsets/GLUE/mnli_mismatched.json",
    "qnli": "/home/sichenglei/PromptQA/testsets/GLUE/qnli.json",
    "qqp": "/home/sichenglei/PromptQA/testsets/GLUE/qqp.json",
    "rte": "/home/sichenglei/PromptQA/testsets/GLUE/rte.json",
    "sst2": "/home/sichenglei/PromptQA/testsets/GLUE/sst2.json",
    "mrpc": "/home/sichenglei/PromptQA/testsets/GLUE/qqp_to_mrpc.json",
    "snli": "/home/sichenglei/PromptQA/testsets/GLUE/snli.json",
    "rte": "/home/sichenglei/PromptQA/testsets/GLUE/rte.json",
    "scitail": "/home/sichenglei/PromptQA/testsets/GLUE/scitail.json",
    "qnli": "/home/sichenglei/PromptQA/testsets/GLUE/qnli.json",
    "wnli": "/home/sichenglei/PromptQA/testsets/GLUE/wnli.json",
}

advglue_subset_mappings = {
    "mnli": "/home/sichenglei/PromptQA/testsets/AdvGLUE/mnli_matched.json",
    "mnli-mm":  "/home/sichenglei/PromptQA/testsets/AdvGLUE/mnli_mismatched.json",
    "qnli": "/home/sichenglei/PromptQA/testsets/AdvGLUE/qnli.json",
    "qqp": "/home/sichenglei/PromptQA/testsets/AdvGLUE/qqp.json",
    "rte": "/home/sichenglei/PromptQA/testsets/AdvGLUE/rte.json",
    "sst2": "/home/sichenglei/PromptQA/testsets/AdvGLUE/sst2.json",
}

nli_train_mappings = {
    "mnli": "/home/sichenglei/AdvGLUE/GLUE/MNLI/train.tsv",
    "mnli-mm": "/home/sichenglei/AdvGLUE/GLUE/MNLI/train.tsv",
    "rte": "/home/sichenglei/LM-BFF/data/original/RTE/train.tsv",
    "scitail": "/home/sichenglei/MoreNLI/SciTail/train.tsv",
    "qnli": "/home/sichenglei/MoreNLI/QNLI/train.tsv",
    "wnli": "/home/sichenglei/MoreNLI/WNLI/train.tsv",
    "qqp": "/home/sichenglei/AdvGLUE/GLUE/QQP/train.tsv",
    "mrpc": "/home/sichenglei/LM-BFF/data/original/MRPC/train.tsv",
}

nli_test_mappings = {
    "mnli": "/home/sichenglei/AdvGLUE/GLUE/MNLI/dev_matched.tsv",
    "mnli-mm": "/home/sichenglei/AdvGLUE/GLUE/MNLI/dev_mismatched.tsv",
    "rte": "/home/sichenglei/LM-BFF/data/original/RTE/dev.tsv",
    "scitail": "/home/sichenglei/MoreNLI/SciTail/dev.tsv",
    "qnli": "/home/sichenglei/MoreNLI/QNLI/dev.tsv",
    "wnli": "/home/sichenglei/MoreNLI/WNLI/dev.tsv",
    "qqp": "/home/sichenglei/AdvGLUE/GLUE/QQP/dev.tsv",
    "mrpc": "/home/sichenglei/LM-BFF/data/original/MRPC/dev.tsv",
}

nli_subset_mappings = {
    "mnli": "/home/sichenglei/PromptQA/testsets/NLI_OOD/mnli.json",
    "mnli-mm": "/home/sichenglei/PromptQA/testsets/NLI_OOD/mnli_mm.json",
    "rte": "/home/sichenglei/PromptQA/testsets/NLI_OOD/rte.json",
    "scitail": "/home/sichenglei/PromptQA/testsets/NLI_OOD/scitail.json",
    "qnli": "/home/sichenglei/PromptQA/testsets/NLI_OOD/qnli.json",
    "wnli": "/home/sichenglei/PromptQA/testsets/NLI_OOD/wnli.json",
    "qqp": "/home/sichenglei/PromptQA/testsets/NLI_OOD/qqp.json",
    "mrpc": "/home/sichenglei/PromptQA/testsets/NLI_OOD/mrpc.json",
}

def sample_strategyqa(task="strategyqa"):
    with open("/home/sichenglei/QAdatasets/strategyqa.json", "r") as f:
        data = json.load(f)
    
    olddata = data["examples"]
    random.shuffle(olddata)
    ## reformat explanations a little bit 
    data = []
    for i in range(len(olddata)):
        data.append({})
        data[-1]["question"] = olddata[i]["input"]
        if olddata[i]["target_scores"]["Yes"] == 1:
            data[-1]["answer"] = ["yes"]
        else:
            data[-1]["answer"] = ["no"]
        cot = olddata[i]["target"].split(". ")
        # cot = ". ".join(cot[1 : ]) + " Therefore the answer is " + cot[0].lower() + "."
        cot = ". ".join(cot[1 : ])
        data[-1]["cot"] = cot
    print ("#orig train data: ", len(data))

    newdata = {}
    demo_count = 16
    newdata["dataset"] = task
    newdata["demos"] = data[ : demo_count]
    newdata["testset"] = data[demo_count : ]

    print ("#shots: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))

    with open("/home/sichenglei/PromptQA/testsets/strategyqa.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample(task="nq"):
    print ("processing: ", task)
    newdata = {}
    newdata["dataset"] = task
    newdata["demos"] = []
    newdata["testset"] = []

    if task in ["qampari", "boolq"] or "timeqa" in task:
        data = []
        with open(train_mappings[task]) as f:
            for line in f:
                data.append(json.loads(line))
    elif task == "race":
        data = []
        files = []
        for dir in train_mappings[task]:
            files.extend([join(dir, f) for f in listdir(dir) if isfile(join(dir, f))])
        # print ("#race files: ", len(files))
        for f in files:
            with open(f, "r") as ff:
                d = json.load(ff)
            data.append(d)
    elif task == "boolq-rc":
        data = []
        with open(train_mappings[task]) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                qc = row[0].split("\\n")
                ans = row[1]
                q = qc[0].strip()
                c = qc[1].strip()
                data.append([c, q, ans])
    elif "mrqa-" in task:
        data = []
        with open(train_mappings[task]) as f:
            json_list = list(f)
            for i,line in enumerate(json_list):
                if i == 0:
                    continue 
                data.append(json.loads(line))
    else:
        with open(train_mappings[task], "r") as f:
            data = json.load(f)
    if task in ["webqsp" ,"freebaseqa"]:
        data = data["Questions"]
    
    print ("#orig train data: ", len(data))
    if "subqa" not in task:
        random.shuffle(data)

    if "subqa" not in task:
        demo_count = 1024
    else:
        demo_count = 16 

    if task == "hotpotqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            if type(data[i]["answer"]) is not list:
                neweg["answer"] = [data[i]["answer"]]
            else:
                neweg["answer"] = data[i]["answer"]
            newdata["demos"].append(neweg)
    elif task == "webq":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qId"]
            neweg["question"] = data[i]["qText"]
            if type(data[i]["answers"]) is not list:
                neweg["answer"] = [data[i]["answers"]]
            else:
                neweg["answer"] = data[i]["answers"]
            newdata["demos"].append(neweg)
    elif task == "cwq":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["ID"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answers"]:
                neweg["answer"].append(ans["answer"])
                neweg["answer"] += ans["aliases"]
            newdata["demos"].append(neweg)
    elif task == "qampari":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question_text"]
            neweg["answer"] = []
            for ans in data[i]["answer_list"]:
                neweg["answer"].append([])
                neweg["answer"][-1].append(ans["answer_text"])
                neweg["answer"][-1] += ans["aliases"]
            newdata["demos"].append(neweg)
    elif task == "ambigqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["annotations"] = data[i]["annotations"]
            neweg["answer"] = []
            for dq in data[i]["annotations"]:
                if dq["type"] == "singleAnswer":
                    neweg["answer"].append(dq["answer"])
                else:
                    for qa in dq["qaPairs"]:
                        neweg["answer"].append(qa["answer"])
            newdata["demos"].append(neweg)
    elif task == "boolq":
        for i in range(demo_count):
            neweg = {}
            # neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [str(data[i]["answer"])]
            newdata["demos"].append(neweg)
    elif task == "webqsp":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["QuestionId"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for ans in data[i]["Parses"]:
                for a in ans["Answers"]:
                    neweg["answer"].append(a['EntityName'])
            ## filter out no-answer questions
            if len(neweg["answer"]) == 0 or neweg["answer"][0] is None:
                continue 
            newdata["demos"].append(neweg)
    elif "timeqa" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["idx"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = data[i]["targets"]
            newdata["demos"].append(neweg)
    elif task == "grailqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answer"]:
                if ans["answer_type"] == "Entity":
                    neweg["answer"].append(ans["entity_name"])
                elif ans["answer_type"] == "Value":
                    neweg["answer"].append(ans["answer_argument"])
            newdata["demos"].append(neweg)
    elif "cfq-mcd" in task:
        for i in range(demo_count):
            neweg = {}
            # neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["expectedResponse"]]
            newdata["demos"].append(neweg)
    elif "freebaseqa" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["Question-ID"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for p in data[i]["Parses"]:
                for ans in p["Answers"]:
                    neweg["answer"].extend(ans["AnswersName"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["demos"].append(neweg)
    elif task == "hybridqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["question_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer-text"]]
            newdata["demos"].append(neweg)
    elif task == "reclor":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["id_string"]
            neweg["question"] = data[i]["context"]
            neweg["question"] += "\n" + data[i]["question"] + "\n"
            for a in range(len(data[i]["answers"])):
                letter = chr(ord("A") + a)  + ". "
                neweg["question"] += letter + data[i]["answers"][a] + "\n"
            neweg["answer"] = [chr(ord("A") + int(data[i]["label"]))]
            newdata["demos"].append(neweg)
    elif task == "race":
        for i in range(demo_count):
            for q in range(len(data[i]["answers"])):
                neweg = {}
                neweg["id"] = data[i]["id"]
                neweg["question"] = data[i]["article"]
                neweg["question"] += "\n" + data[i]["questions"][q] + "\n"
                for a in range(len(data[i]["options"][q])):
                    letter = chr(ord("A") + a)  + ". "
                    neweg["question"] += letter + data[i]["options"][q][a] + "\n"
                neweg["answer"] = [data[i]["answers"][q]]

                newdata["demos"].append(neweg)
    elif task == "boolq-rc":
        for i in range(demo_count):
            neweg = {}
            neweg["question"] = data[i][0] + "\n" + data[i][1]
            neweg["answer"] = [data[i][2].strip()]
            newdata["demos"].append(neweg)
    elif task == "subqa-overall-passage":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            support = [d[0] for d in data[i]["supporting_facts"]]
            passages = [d[1] for d in data[i]["context"] if d[0] in support]
            passages = [' '.join(d) for d in passages]
            passage = '\n'.join(passages) + '\n'

            neweg["question"] = passage + data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["demos"].append(neweg)
    elif task == "subqa-all10":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["demos"].append(neweg)
    elif "subqa" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["demos"].append(neweg)
    elif "mrqa-" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["uid"]
            neweg["question"] = data[i]["query"]
            neweg["answer"] = []
            for ans in data[i]["gold_answers"]:
                neweg["answer"].append(ans["text"])
                if ans["aliases"] != None:
                    neweg["answer"].extend(ans["aliases"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["demos"].append(neweg)
    else:
        newdata["demos"] = data[ : demo_count]

    print ("#sampled demo from train: ", len(newdata["demos"]))

    
    
    
    if task in ["qampari", "boolq"] or "timeqa" in task:
        data = []
        with open(test_mappings[task]) as f:
            for line in f:
                data.append(json.loads(line))
    elif task == "race":
        data = []
        files = []
        for dir in test_mappings[task]:
            files.extend([join(dir, f) for f in listdir(dir) if isfile(join(dir, f))])
        for f in files:
            with open(f, "r") as ff:
                d = json.load(ff)
            data.append(d)
    elif task == "boolq-rc":
        data = []
        with open(test_mappings[task]) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                qc = row[0].split("\\n")
                ans = row[1]
                q = qc[0].strip()
                c = qc[1].strip()
                data.append([c, q, ans])
    elif ("subqa" not in task) and ("mrqa-" not in task):
        with open(test_mappings[task], "r") as f:
            data = json.load(f)
    
    if task in ["webqsp" ,"freebaseqa"]:
        data = data["Questions"]
    
    if ("subqa" in task):
        data = data[demo_count : ]
    elif ("mrqa-" in task): 
        data = data[demo_count : demo_count + 25000]
        
    if task == "hotpotqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            if type(data[i]["answer"]) is not list:
                neweg["answer"] = [data[i]["answer"]]
            else:
                neweg["answer"] = data[i]["answer"]
            newdata["testset"].append(neweg)
    elif task == "webq":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qId"]
            neweg["question"] = data[i]["qText"]
            if type(data[i]["answers"]) is not list:
                neweg["answer"] = [data[i]["answers"]]
            else:
                neweg["answer"] = data[i]["answers"]
            newdata["testset"].append(neweg)
    elif task == "cwq":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["ID"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answers"]:
                neweg["answer"].append(ans["answer"])
                neweg["answer"] += ans["aliases"]
            newdata["testset"].append(neweg)
    elif task == "qampari":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question_text"]
            neweg["answer"] = []
            for ans in data[i]["answer_list"]:
                neweg["answer"].append([])
                neweg["answer"][-1].append(ans["answer_text"])
                neweg["answer"][-1] += ans["aliases"]
            newdata["testset"].append(neweg)
    elif task == "ambigqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["annotations"] = data[i]["annotations"]
            neweg["answer"] = []
            for dq in data[i]["annotations"]:
                if dq["type"] == "singleAnswer":
                    neweg["answer"].append(dq["answer"])
                else:
                    for qa in dq["qaPairs"]:
                        neweg["answer"].append(qa["answer"])
            newdata["testset"].append(neweg)
    elif task == "boolq":
        for i in range(len(data)):
            neweg = {}
            # neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [str(data[i]["answer"])]
            newdata["testset"].append(neweg)
    elif task == "webqsp":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["QuestionId"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for ans in data[i]["Parses"]:
                for a in ans["Answers"]:
                    neweg["answer"].append(a['EntityName'])
            if len(neweg["answer"]) == 0 or neweg["answer"][0] is None:
                continue 
            newdata["testset"].append(neweg)
    elif "timeqa" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["idx"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = data[i]["targets"]
            newdata["testset"].append(neweg)
    elif task == "grailqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            neweg["level"] = data[i]["level"]
            for ans in data[i]["answer"]:
                if ans["answer_type"] == "Entity":
                    neweg["answer"].append(ans["entity_name"])
                elif ans["answer_type"] == "Value":
                    neweg["answer"].append(ans["answer_argument"])
            newdata["testset"].append(neweg)
    elif "cfq-mcd" in task:
        for i in range(len(data)):
            neweg = {}
            # neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["expectedResponse"]]
            newdata["testset"].append(neweg)
    elif "freebaseqa" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["Question-ID"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for p in data[i]["Parses"]:
                for ans in p["Answers"]:
                    neweg["answer"].extend(ans["AnswersName"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["testset"].append(neweg)
    elif task == "hybridqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["question_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer-text"]]
            newdata["testset"].append(neweg)
    elif task == "reclor":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["id_string"]
            neweg["question"] = data[i]["context"]
            neweg["question"] += "\n" + data[i]["question"] + "\n"
            for a in range(len(data[i]["answers"])):
                letter = chr(ord("A") + a)  + ". "
                neweg["question"] += letter + data[i]["answers"][a] + "\n"
            neweg["answer"] = [chr(ord("A") + int(data[i]["label"]))]
            newdata["testset"].append(neweg)
    elif task == "race":
        for i in range(len(data)):
            for q in range(len(data[i]["answers"])):
                neweg = {}
                neweg["id"] = data[i]["id"]
                neweg["question"] = data[i]["article"]
                neweg["question"] += "\n" + data[i]["questions"][q] + "\n"
                for a in range(len(data[i]["options"][q])):
                    letter = chr(ord("A") + a)  + ". "
                    neweg["question"] += letter + data[i]["options"][q][a] + "\n"
                neweg["answer"] = [data[i]["answers"][q]]

                newdata["testset"].append(neweg)
    elif task == "boolq-rc":
        for i in range(len(data)):
            neweg = {}
            neweg["question"] = data[i][0] + "\n" + data[i][1]
            neweg["answer"] = [data[i][2].strip()]
            newdata["testset"].append(neweg)
    elif task == "subqa-overall-passage":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            support = [d[0] for d in data[i]["supporting_facts"]]
            passages = [d[1] for d in data[i]["context"] if d[0] in support]
            passages = [' '.join(d) for d in passages]
            passage = '\n'.join(passages) + '\n'

            neweg["question"] = passage + data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["testset"].append(neweg)
    elif task == "subqa-all10":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            passages = [d[1] for d in data[i]["context"]]
            passages = [' '.join(d) for d in passages]
            passage = '\n'.join(passages) + '\n'

            neweg["question"] = passage + data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["testset"].append(neweg)
    elif "subqa" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer"]]
            newdata["testset"].append(neweg)
    elif "mrqa-" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["uid"]
            neweg["question"] = data[i]["query"]
            neweg["answer"] = []
            for ans in data[i]["gold_answers"]:
                neweg["answer"].append(ans["text"])
                if ans["aliases"] != None:
                    neweg["answer"].extend(ans["aliases"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["testset"].append(neweg)
    else:
        newdata["testset"] = data
    
    print ("#test data: ", len(newdata["testset"]))

    with open(subset_mappings[task], "w") as f:
        json.dump(newdata, f, indent=4)


def sample_train(task="nq"):
    print ("processing: ", task)
    newdata = {}
    newdata["dataset"] = task
    newdata["demos"] = []
    newdata["testset"] = []

   
    with open(train_mappings[task], "r") as f:
        data = json.load(f)
    if task in ["webqsp" ,"freebaseqa"]:
        data = data["Questions"]
    
    print ("#orig train data: ", len(data))
    random.shuffle(data)

    demo_count = 1024
    if task == "hotpotqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            if type(data[i]["answer"]) is not list:
                neweg["answer"] = [data[i]["answer"]]
            else:
                neweg["answer"] = data[i]["answer"]
            newdata["demos"].append(neweg)
    elif task == "webq":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qId"]
            neweg["question"] = data[i]["qText"]
            if type(data[i]["answers"]) is not list:
                neweg["answer"] = [data[i]["answers"]]
            else:
                neweg["answer"] = data[i]["answers"]
            newdata["demos"].append(neweg)
    elif task == "cwq":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["ID"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answers"]:
                neweg["answer"].append(ans["answer"])
                neweg["answer"] += ans["aliases"]
            newdata["demos"].append(neweg)
    elif task == "qampari":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question_text"]
            neweg["answer"] = []
            for ans in data[i]["answer_list"]:
                neweg["answer"].append([])
                neweg["answer"][-1].append(ans["answer_text"])
                neweg["answer"][-1] += ans["aliases"]
            newdata["demos"].append(neweg)
    elif task == "ambigqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["annotations"] = data[i]["annotations"]
            neweg["answer"] = []
            for dq in data[i]["annotations"]:
                if dq["type"] == "singleAnswer":
                    neweg["answer"].append(dq["answer"])
                else:
                    for qa in dq["qaPairs"]:
                        neweg["answer"].append(qa["answer"])
            newdata["demos"].append(neweg)
    elif task == "boolq":
        for i in range(demo_count):
            neweg = {}
            # neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [str(data[i]["answer"])]
            newdata["demos"].append(neweg)
    elif task == "webqsp":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["QuestionId"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for ans in data[i]["Parses"]:
                for a in ans["Answers"]:
                    neweg["answer"].append(a['EntityName'])
            ## filter out no-answer questions
            if len(neweg["answer"]) == 0 or neweg["answer"][0] is None:
                continue 
            newdata["demos"].append(neweg)
    elif "timeqa" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["idx"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = data[i]["targets"]
            newdata["demos"].append(neweg)
    elif task == "grailqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answer"]:
                if ans["answer_type"] == "Entity":
                    neweg["answer"].append(ans["entity_name"])
                elif ans["answer_type"] == "Value":
                    neweg["answer"].append(ans["answer_argument"])
            newdata["demos"].append(neweg)
    elif "cfq-mcd" in task:
        for i in range(demo_count):
            neweg = {}
            # neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["expectedResponse"]]
            newdata["demos"].append(neweg)
    elif "freebaseqa" in task:
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["Question-ID"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for p in data[i]["Parses"]:
                for ans in p["Answers"]:
                    neweg["answer"].extend(ans["AnswersName"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["demos"].append(neweg)
    elif task == "hybridqa":
        for i in range(demo_count):
            neweg = {}
            neweg["id"] = data[i]["question_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer-text"]]
            newdata["demos"].append(neweg)
    else:
        newdata["demos"] = data[ : demo_count]

    print ("#sampled demo from train: ", len(newdata["demos"]))

    
    # data = data[demo_count + 25000 : ]
    with open(test_mappings[task], "r") as f:
        data = json.load(f)
    
    if task == "hotpotqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["_id"]
            neweg["question"] = data[i]["question"]
            if type(data[i]["answer"]) is not list:
                neweg["answer"] = [data[i]["answer"]]
            else:
                neweg["answer"] = data[i]["answer"]
            newdata["testset"].append(neweg)
    elif task == "webq":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qId"]
            neweg["question"] = data[i]["qText"]
            if type(data[i]["answers"]) is not list:
                neweg["answer"] = [data[i]["answers"]]
            else:
                neweg["answer"] = data[i]["answers"]
            newdata["testset"].append(neweg)
    elif task == "cwq":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["ID"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            for ans in data[i]["answers"]:
                neweg["answer"].append(ans["answer"])
                neweg["answer"] += ans["aliases"]
            newdata["testset"].append(neweg)
    elif task == "qampari":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question_text"]
            neweg["answer"] = []
            for ans in data[i]["answer_list"]:
                neweg["answer"].append([])
                neweg["answer"][-1].append(ans["answer_text"])
                neweg["answer"][-1] += ans["aliases"]
            newdata["testset"].append(neweg)
    elif task == "ambigqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["annotations"] = data[i]["annotations"]
            neweg["answer"] = []
            for dq in data[i]["annotations"]:
                if dq["type"] == "singleAnswer":
                    neweg["answer"].append(dq["answer"])
                else:
                    for qa in dq["qaPairs"]:
                        neweg["answer"].append(qa["answer"])
            newdata["testset"].append(neweg)
    elif task == "boolq":
        for i in range(len(data)):
            neweg = {}
            # neweg["id"] = data[i]["id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [str(data[i]["answer"])]
            newdata["testset"].append(neweg)
    elif task == "webqsp":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["QuestionId"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for ans in data[i]["Parses"]:
                for a in ans["Answers"]:
                    neweg["answer"].append(a['EntityName'])
            if len(neweg["answer"]) == 0 or neweg["answer"][0] is None:
                continue 
            newdata["testset"].append(neweg)
    elif "timeqa" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["idx"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = data[i]["targets"]
            newdata["testset"].append(neweg)
    elif task == "grailqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = []
            neweg["level"] = data[i]["level"]
            for ans in data[i]["answer"]:
                if ans["answer_type"] == "Entity":
                    neweg["answer"].append(ans["entity_name"])
                elif ans["answer_type"] == "Value":
                    neweg["answer"].append(ans["answer_argument"])
            newdata["testset"].append(neweg)
    elif "cfq-mcd" in task:
        for i in range(len(data)):
            neweg = {}
            # neweg["id"] = data[i]["qid"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["expectedResponse"]]
            newdata["testset"].append(neweg)
    elif "freebaseqa" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["Question-ID"]
            neweg["question"] = data[i]["ProcessedQuestion"]
            neweg["answer"] = []
            for p in data[i]["Parses"]:
                for ans in p["Answers"]:
                    neweg["answer"].extend(ans["AnswersName"])
            neweg["answer"] = list(set(neweg["answer"]))
            newdata["testset"].append(neweg)
    elif task == "hybridqa":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i]["question_id"]
            neweg["question"] = data[i]["question"]
            neweg["answer"] = [data[i]["answer-text"]]
            newdata["testset"].append(neweg)
    else:
        newdata["testset"] = data
    
    print ("#test data: ", len(newdata["testset"]))

    with open(train_for_inference_mappings[task], "w") as f:
        json.dump(newdata, f, indent=4)


def sample_nq_dpr():
    with open("/home/sichenglei/PromptQA/testsets/nq.json", "r") as f:
        orig = json.load(f)
    
    newdata = {}
    newdata["dataset"] = orig["dataset"]
    newdata["demos"] = orig["demos"]
    newdata["testset"] = []
    orig_questions = {} 
    for qd in orig["testset"]:
        orig_questions[qd["question"]] = qd
    
    had = []

    with open("/data3/private/clsi/UnifiedQAdata/natural_questions_with_dpr_para/test.tsv", "r") as f:
        dpr = csv.reader(f, delimiter="\t", quotechar='"')
    
        for row in dpr:
            qc = row[0].split("\\n")
            q = qc[0]
            c = qc[1]
            ans = row[1]

            if q[:-1] in orig_questions and q not in had:
                had.append(q)

                neweg = {}
                neweg["id"] = orig_questions[q[:-1]]["id"]
                neweg["question"] = c + '\n' + q 
                neweg["answer"] = orig_questions[q[:-1]]["answer"]
                newdata["testset"].append(neweg)
    
    print (len(newdata["testset"]))

    with open("/home/sichenglei/PromptQA/testsets/nq-dpr.json", "w") as f:
        json.dump(newdata, f, indent=4)
            

def sample_glue(task="mnli"):
    print ("processing: ", task)
    newdata = {}
    newdata["dataset"] = task
    newdata["demos"] = []
    newdata["testset"] = []
 
    ## Use QQP demos
    if task == "mrpc":
        with open(glue_train_mappings["qqp"]) as fd:
            data = list(csv.reader(fd, delimiter="\t"))[1 : ]
    else:
        with open(glue_train_mappings[task]) as fd:
            data = list(csv.reader(fd, delimiter="\t"))[1 : ]
    random.shuffle(data)
    data = data[ : 1000]

    demo_count = 16 
    if "mnli" in task:
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][2]
            neweg["question"] = demos[i][8].replace('\n', ' ').strip() + '\n' + demos[i][9].replace('\n', ' ').strip() + '\n' + "What is the relationship between these two sentences?"
            neweg["answer"] = demos[i][-1]
            newdata["demos"].append(neweg)

    elif task == "qnli":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][2].replace('\n', ' ').strip() + '\n' + demos[i][1].replace('\n', ' ').strip() + '\n' + "Is the answer to the question entailed in the given context?"
            label = demos[i][-1].strip()
            if label == "entailment":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "qqp":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][3].replace('\n', ' ').strip() + '\n' + demos[i][4].replace('\n', ' ').strip() + '\n' + "Are these two questions asking the same thing?"
            label = demos[i][-1].strip()
            if label == "1":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "rte":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][1].replace('\n', ' ').strip() + '\n' + demos[i][2].replace('\n', ' ').strip() + '\n' + "Can the second sentence be inferred from the first sentence?"
            label = demos[i][-1].strip()
            if label == "entailment":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "sst2":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = demos[i][0].replace('\n', ' ').strip() + '\n' + "What is the sentiment of this sentence?"
            label = demos[i][-1].strip()
            if label == "1":
                label = "positive"
            else:
                label = "negative"
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "mrpc":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][3].replace('\n', ' ').strip() + '\n' + demos[i][4].replace('\n', ' ').strip() + '\n' + "Are these two sentences paraphrases?"
            label = demos[i][-1].strip()
            if label == "1":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    if "snli" in task:
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][7].replace('\n', ' ').strip() + '\n' + demos[i][8].replace('\n', ' ').strip()
            neweg["answer"] = demos[i][-1]
            newdata["demos"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))

    
    ## cap the test size to 10k
    with open(glue_test_mappings[task]) as fd:
        data = list(csv.reader(fd, delimiter="\t"))[1 : 10000]

    if "mnli" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][2]
            neweg["question"] = data[i][8].replace('\n', ' ').strip() + '\n' + data[i][9].replace('\n', ' ').strip() + '\n' + "What is the relationship between these two sentences?"
            neweg["answer"] = data[i][-1]
            newdata["testset"].append(neweg)

            # neweg = {}
            # neweg["id"] = data[i][2]
            # senta = data[i][8].replace('\n', ' ').strip()
            # sentb = data[i][9].replace('\n', ' ').strip()
            # if sentb[-1] in puncs:
            #     sentb = sentb[ : -1] + '?'
            # else:
            #     sentb += '?'
            # label = data[i][-1].strip().lower()
            # if label == "entailment":
            #     label = "yes"
            # elif label == "contradiction":
            #     label = "no"
            # else:
            #     label = "maybe"
            # neweg["question"] = senta + ' ' + sentb
            # neweg["answer"] = label
            # newdata["testset"].append(neweg)

    elif task == "qnli":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][2].replace('\n', ' ').strip() + '\n' + data[i][1].replace('\n', ' ').strip() + '\n' + "Is the answer to the question entailed in the given context?"
            label = data[i][-1].strip()
            if label == "entailment":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "qqp":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][3].replace('\n', ' ').strip() + '\n' + data[i][4].replace('\n', ' ').strip() + '\n' + "Are these two questions asking the same thing?"
            label = data[i][-1].strip()
            if label == "1":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "rte":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][1].replace('\n', ' ').strip() + '\n' + data[i][2].replace('\n', ' ').strip() + '\n' + "Can the second sentence be inferred from the first sentence?"
            label = data[i][-1].strip()
            if label == "entailment":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "sst2":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = data[i][0].replace('\n', ' ').strip() + '\n' + "What is the sentiment of this sentence?"
            label = data[i][-1].strip()
            if label == "1":
                label = "positive"
            else:
                label = "negative"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "mrpc":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = str(i)
            if len(data[i]) != 5:
                a = data[i][-1].split("\t")
                data[i] = data[i][ : -1 ] + a
            neweg["question"] = data[i][3].replace('\n', ' ').strip() + '\n' + data[i][4].replace('\n', ' ').strip() + '\n' + "Are these two sentences paraphrases?"
            label = data[i][0].strip()
            if label == "1":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "snli":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = data[i][7].replace('\n', ' ').strip() + '\n' + data[i][8].replace('\n', ' ').strip()
            neweg["answer"] = data[i][-1]
            newdata["testset"].append(neweg)

    
    print ("#test data: ", len(newdata["testset"]))

    with open(glue_subset_mappings[task], "w") as f:
        json.dump(newdata, f, indent=4)


def sample_adv_glue(task="sst2"):
    source = "/home/sichenglei/AdvGLUE/dev/dev.json"
    with open(source, "r") as f:
        data = json.load(f)
    newdata = {}
    newdata["dataset"] = task
    newdata["demos"] = []
    newdata["testset"] = []

    ## load orig demos
    with open(glue_subset_mappings[task]) as f:
        orig_data = json.load(f)
    newdata["demos"] = orig_data["demos"]

    data = data[task]
    if task == "sst2":
        for line in data:
            neweg = {}
            neweg["id"] = line["idx"]
            neweg["question"] = line["sentence"].replace('\n', ' ').strip() + '\n' + "What is the sentiment of this sentence?"
            label = str(line["label"])
            if label == "1":
                label = "positive"
            else:
                label = "negative"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    elif task == "rte":
        for line in data:
            neweg = {}
            neweg["id"] = line["idx"]
            neweg["question"] = line["sentence1"].replace('\n', ' ').strip() + '\n' + line["sentence2"].replace('\n', ' ').strip() + '\n' + "Can the second sentence be inferred from the first sentence?"
            label = str(line["label"])
            if label == "0":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    elif task == "qnli":
        for line in data:
            neweg = {}
            neweg["id"] = line["idx"]
            neweg["question"] = line["sentence"].replace('\n', ' ').strip() + '\n' + line["question"].replace('\n', ' ').strip() + '\n' + "Is the answer to the question entailed in the given context?"
            label = str(line["label"])
            if label == "0":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif "mnli" in task:
        for line in data:
            neweg = {}
            neweg["id"] = line["idx"]
            neweg["question"] = line["premise"].replace('\n', ' ').strip() + '\n' + line["hypothesis"].replace('\n', ' ').strip() + '\n' + "What is the relationship between these two sentences?"
            label = str(line["label"])
            if label == "1":
                label = "neutral"
            elif label == "0":
                label = "entailment"
            else:
                label = "contradiction"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "qqp":
        for line in data:
            neweg = {}
            neweg["id"] = line["idx"]
            neweg["question"] = line["question1"].replace('\n', ' ').strip() + '\n' + line["question2"].replace('\n', ' ').strip() + '\n' + "Are these two questions asking the same thing?"
            label = str(line["label"])
            if label == "1":
                label = "yes"
            else:
                label = "no"
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    
    
    print ("#test data: ", len(newdata["testset"]))

    with open(advglue_subset_mappings[task], "w") as f:
        json.dump(newdata, f, indent=4)


def sample_contrast_imdb():
    demo_count = 8
    # with open("/home/sichenglei/contrast-sets/IMDb/data/test_original.tsv", "r") as fd:
    #     data = []
    #     for i, line in enumerate(fd.readlines()):
    #         if i == 0:
    #             continue 
    #         data.append(line.split("\t"))
        
    #     group_by_label = {}
    #     for i in range(len(data)):
    #         label = data[i][0].strip().lower()
    #         if label not in group_by_label:
    #             group_by_label[label] = [data[i]]
    #         else:
    #             group_by_label[label].append(data[i])
        
    #     demos = []
    #     testset = []
    #     for k,v in group_by_label.items():
    #         demos.extend(v[ : demo_count])
    #         testset.extend(v[demo_count : ])
    #     random.shuffle(demos)
        
    #     newdata = {}
    #     newdata["dataset"] = "imdb"
    #     newdata["demos"] = []
    #     newdata["testset"] = []

    #     for d in demos:
    #         neweg = {}
    #         neweg["question"] = d[1].replace("\n", ' ').strip() + '\n' + "What is the sentiment of this sentence?"    
    #         neweg["answer"] = d[0].strip().lower()
    #         newdata["demos"].append(neweg)
        
    #     for d in testset:
    #         neweg = {}
    #         neweg["question"] = d[1].replace("\n", ' ').strip() + '\n' + "What is the sentiment of this sentence?"    
    #         neweg["answer"] = d[0].strip().lower()
    #         newdata["testset"].append(neweg)
    
    # print ("#demos: ", len(newdata["demos"]))
    # print ("#test data: ", len(newdata["testset"]))
    # with open("/home/sichenglei/PromptQA/testsets/ContrastSet/imdb_orig.json", "w") as f:
    #     json.dump(newdata, f, indent=4)

    with open("/home/sichenglei/contrast-sets/IMDb/data/test_contrast.tsv", "r") as fd:
        data = []
        for i, line in enumerate(fd.readlines()):
            if i == 0:
                continue 
            data.append(line.split("\t"))
        
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][0].strip().lower()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        
        demos = []
        testset = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
            testset.extend(v[demo_count : ])
        random.shuffle(demos)
        
        newdata = {}
        newdata["dataset"] = "imdb"
        newdata["demos"] = []
        newdata["testset"] = []

        # for d in demos:
        #     neweg = {}
        #     neweg["question"] = d[1].replace("\n", ' ').strip() + '\n' + "What is the sentiment of this sentence?"    
        #     neweg["answer"] = d[0].strip().lower()
        #     newdata["demos"].append(neweg)

        with open("/home/sichenglei/PromptQA/testsets/ContrastSet/imdb_orig.json", "r") as f:
            imdb_orig = json.load(f)
        newdata["demos"] = imdb_orig["demos"]
        
        for d in testset:
            neweg = {}
            neweg["question"] = d[1].replace("\n", ' ').strip() + '\n' + "What is the sentiment of this sentence?"    
            neweg["answer"] = d[0].strip().lower()
            newdata["testset"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))
    print ("#test data: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/ContrastSet/imdb_contrast.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_contrast_boolq():
    demo_count = 8
    with open("/home/sichenglei/PromptQA/testsets/AllQA/boolq-rc.json", "r") as fd:
        boolq_orig = json.load(fd)
        demos = boolq_orig["demos"]    
    
    group_by_label = {}
    for i in range(len(demos)):
        label = demos[i]["answer"][0]
        if label not in group_by_label:
            group_by_label[label] = [demos[i]]
        else:
            group_by_label[label].append(demos[i])
    
    demos = []
    for k,v in group_by_label.items():
        demos.extend(v[ : demo_count])
    random.shuffle(demos)
    
    newdata = {}
    newdata["dataset"] = "boolq"
    newdata["demos"] = demos 
    newdata["testset"] = []

    with open("/home/sichenglei/contrast-sets/BoolQ/boolq_perturbed.json", "r") as fd:
        data = json.load(fd)
        data = data["data"][ 1 : ]

    # for i,d in enumerate(data):
    #     neweg = {}
    #     neweg["id"] = str(i)
    #     neweg["question"] = d["paragraph"].replace("\n", ' ').strip() + '\n' + d["question"]
    #     if d["answer"] == "FALSE":
    #         neweg["answer"] = ["no"]
    #     else:
    #         neweg["answer"] = ["yes"]
    #     newdata["testset"].append(neweg)

    # print ("#demos: ", len(newdata["demos"]))
    # print ("#test data: ", len(newdata["testset"]))
        
    # with open("/home/sichenglei/PromptQA/testsets/ContrastSet/boolq_orig.json", "w") as f:
    #     json.dump(newdata, f, indent=4)

    for i,d in enumerate(data):
        for dq in d["perturbed_questions"]:
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = d["paragraph"].replace("\n", ' ').strip() + '\n' + dq["perturbed_q"]
            if dq["answer"] == "FALSE":
                neweg["answer"] = ["no"]
            else:
                neweg["answer"] = ["yes"]
            newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#test data: ", len(newdata["testset"]))
        
    with open("/home/sichenglei/PromptQA/testsets/ContrastSet/boolq_contrast.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_contrast_quoref():
    demo_count = 16
    with open("/home/sichenglei/QAdatasets/quoref-train-dev-v0.1/quoref-train-v0.1.json", "r") as fd:
        orig = json.load(fd)
        data = orig["data"]    
    
    demos = []
    for dq in data:
        for para in dq["paragraphs"]:
            passage = para["context"].replace("\n", ' ').strip()
            for qa in para["qas"]:
                neweg = {}
                neweg["id"] = qa["id"]
                neweg["question"] = passage + '\n' + qa["question"]
                neweg["answer"] = []
                for ans in qa["answers"]:
                    neweg["answer"].append(ans["text"])
                demos.append(neweg)
    
    random.shuffle(demos)
    demos = demos[ : demo_count]
    
    newdata = {}
    newdata["dataset"] = "quoref"
    newdata["demos"] = demos 
    newdata["testset"] = []

    with open("/home/sichenglei/contrast-sets/quoref/quoref_test_perturbations_20191206_merged.json", "r") as fd:
        data = json.load(fd)
        data = data["data"]

    for dq in data:
        for para in dq["paragraphs"]:
            passage = para["context"].replace("\n", ' ').strip()
            for qa in para["qas"]:
                neweg = {}
                neweg["id"] = qa["id"]
                neweg["question"] = passage + '\n' + qa["question"]
                neweg["answer"] = []
                for ans in qa["answers"]:
                    neweg["answer"].append(ans["text"])
                newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#test data: ", len(newdata["testset"]))
        
    with open("/home/sichenglei/PromptQA/testsets/ContrastSet/quoref_contrast.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_contrast_mctaco():
    demo_count = 16
    data = []
    with open("/home/sichenglei/contrast-sets/MCTACO/mctaco_dev_orig.tsv", "r") as fd:
        lines = fd.readlines()
        for line in lines:
            data.append(line.strip().split("\t"))
    
    demos = []
    for dq in data:
        neweg = {}
        neweg["question"] = dq[0].replace("\n", ' ').strip() + ' ' + dq[1] + '\n' + dq[2] + ". Is that correct?"
        neweg["answer"] = dq[3].strip()
        demos.append(neweg)
    
    random.shuffle(demos)
    demos = demos[ : demo_count]
    
    newdata = {}
    newdata["dataset"] = "mctaco"
    newdata["demos"] = demos 
    newdata["testset"] = []

    data = []
    with open("/home/sichenglei/contrast-sets/MCTACO/changed.tsv", "r") as fd:
        lines = fd.readlines()
        for line in lines[ 1 : ]:
            data.append(line.strip().split("\t"))

    for dq in data:
        neweg = {}
        neweg["id"] = dq[0]
        neweg["question"] = dq[1].replace("\n", ' ').strip() + '\n' + dq[2] + ". Is that correct?"
        neweg["answer"] = dq[3].strip()
        newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#test data: ", len(newdata["testset"]))
        
    with open("/home/sichenglei/PromptQA/testsets/ContrastSet/mctaco_contrast.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_squad_ood():
    with open("/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQASQuADDev.json", "r") as f:
        data = json.load(f)
    demos = data["demos"]
    
    newdata = {}
    newdata["dataset"] = "squad_ood"
    newdata["demos"] = demos 
    newdata["testset"] = []

    with open("/home/sichenglei/QAdatasets/SQuAD_OOD/amazon_reviews_v1.0.json", "r") as fd:
        data = json.load(fd)
        data = data["data"]

    for dq in data:
        for para in dq["paragraphs"]:
            passage = para["context"].replace("\n", ' ').strip()
            for qa in para["qas"]:
                neweg = {}
                neweg["id"] = qa["id"]
                neweg["question"] = passage + '\n' + qa["question"]
                neweg["answer"] = []
                for ans in qa["answers"]:
                    neweg["answer"].append(ans["text"])
                newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#test data: ", len(newdata["testset"]))
        
    with open("/home/sichenglei/PromptQA/testsets/SQuAD_OOD/amazon.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_nli(task="mnli"):
    print ("processing: ", task)
    newdata = {}
    newdata["dataset"] = task
    newdata["demos"] = []
    newdata["testset"] = []
 
    with open(nli_train_mappings[task]) as fd:
        data = list(csv.reader(fd, delimiter="\t"))[1 : ]
    random.shuffle(data)
    data = data[ : 1000]

    demo_count = 16 
    if "mnli" in task:
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][2]
            neweg["question"] = demos[i][8].replace('\n', ' ').strip() + '\n' + demos[i][9].replace('\n', ' ').strip()
            neweg["answer"] = demos[i][-1]
            newdata["demos"].append(neweg)

    elif task == "qnli":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][2].replace('\n', ' ').strip() + '\n' + demos[i][1].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "qqp":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][3].replace('\n', ' ').strip() + '\n' + demos[i][4].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "rte":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][1].replace('\n', ' ').strip() + '\n' + demos[i][2].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "mrpc":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][0].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            if len(demos[i]) != 5:
                continue
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][3].replace('\n', ' ').strip() + '\n' + demos[i][4].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "scitail":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = demos[i][0].replace('\n', ' ').strip() + '\n' + demos[i][1].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    elif task == "wnli":
        group_by_label = {}
        for i in range(len(data)):
            label = data[i][-1].strip()
            if label not in group_by_label:
                group_by_label[label] = [data[i]]
            else:
                group_by_label[label].append(data[i])
        demos = []
        for k,v in group_by_label.items():
            demos.extend(v[ : demo_count])
        random.shuffle(demos)

        for i in range(len(demos)):
            neweg = {}
            neweg["id"] = demos[i][0]
            neweg["question"] = demos[i][1].replace('\n', ' ').strip() + '\n' + demos[i][2].replace('\n', ' ').strip()
            label = demos[i][-1].strip()
            neweg["answer"] = label
            newdata["demos"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))

    
    ## cap the test size to 10k
    with open(nli_test_mappings[task]) as fd:
        data = list(csv.reader(fd, delimiter="\t"))[1 : 10000]

    if "mnli" in task:
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][2]
            neweg["question"] = data[i][8].replace('\n', ' ').strip() + '\n' + data[i][9].replace('\n', ' ').strip()
            neweg["answer"] = data[i][-1]
            newdata["testset"].append(neweg)

    elif task == "qnli":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][2].replace('\n', ' ').strip() + '\n' + data[i][1].replace('\n', ' ').strip()
            label = data[i][-1].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "qqp":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][3].replace('\n', ' ').strip() + '\n' + data[i][4].replace('\n', ' ').strip()
            label = data[i][-1].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "rte":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][1].replace('\n', ' ').strip() + '\n' + data[i][2].replace('\n', ' ').strip()
            label = data[i][-1].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "mrpc":
        for i in range(len(data)):
            if len(data[i]) != 5:
                continue
            neweg = {}
            neweg["id"] = str(i)
            if len(data[i]) != 5:
                a = data[i][-1].split("\t")
                data[i] = data[i][ : -1 ] + a
            neweg["question"] = data[i][3].replace('\n', ' ').strip() + '\n' + data[i][4].replace('\n', ' ').strip()
            label = data[i][0].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "scitail":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = str(i)
            neweg["question"] = data[i][0].replace('\n', ' ').strip() + '\n' + data[i][1].replace('\n', ' ').strip()
            label = data[i][-1].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    elif task == "wnli":
        for i in range(len(data)):
            neweg = {}
            neweg["id"] = data[i][0]
            neweg["question"] = data[i][1].replace('\n', ' ').strip() + '\n' + data[i][2].replace('\n', ' ').strip()
            label = data[i][-1].strip()
            neweg["answer"] = label
            newdata["testset"].append(neweg)
    
    print ("#test data: ", len(newdata["testset"]))

    with open(nli_subset_mappings[task], "w") as f:
        json.dump(newdata, f, indent=4)


def sample_hans():
    newdata = {}
    newdata["dataset"] = "hans"
    newdata["testset"] = []
    lex_ent = []
    lex_non = []
    sub_ent = []
    sub_non = []
    con_ent = []
    con_non = []

    with open("/home/sichenglei/LM-BFF/data/all_data_final/MNLI_HANS/full/test.tsv", "r") as f:
        data = f.readlines()
    
    # print (data[0].split("\t"))
    # print (data[1].split("\t"))
    # data = data[ 1 : ]
    # print (len(data))
    
    # print (len(data[0].split("\t")))

    for line in data:
        line = line.strip().split("\t")
        if len(line) != 14:
            continue
        subtype = line[-4]
        label = line[-1]
        if label == "entailment":
            label = ["entailment"]
        else:
            label = ["neutral", "contradiction"]

        neweg = {}
        neweg["question"] = line[-7].strip().replace("\n", " ") + "\n" + line[-6].strip().replace("\n", " ") + "\n" + "What is the relationship between these two sentences?"
        neweg["answer"] = label

        if subtype[:3] == "lex":
            if "entailment" in label:
                lex_ent.append(neweg)
            else:
                lex_non.append(neweg)
        elif subtype[:3] == "sub":
            if "entailment" in label:
                sub_ent.append(neweg)
            else:
                sub_non.append(neweg)
        elif subtype[:3] == "con":
            if "entailment" in label:
                con_ent.append(neweg)
            else:
                con_non.append(neweg)
    
    # print (len(lex_ent), len(lex_non), len(sub_ent), len(sub_non), len(con_ent), len(con_non))

    random.shuffle(lex_ent)
    lex_ent = lex_ent[ : 1000]
    print ("lex ent: ", len(lex_ent))
    newdata["testset"] = lex_ent
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_lex_ent.json", "w") as f:
        json.dump(newdata, f, indent=4)
    
    random.shuffle(lex_non)
    lex_non = lex_non[ : 1000]
    print ("lex non: ", len(lex_non))
    newdata["testset"] = lex_non
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_lex_non.json", "w") as f:
        json.dump(newdata, f, indent=4)

    random.shuffle(sub_ent)
    sub_ent = sub_ent[ : 1000]
    print ("sub ent: ", len(sub_ent))
    newdata["testset"] = sub_ent
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_sub_ent.json", "w") as f:
        json.dump(newdata, f, indent=4)

    random.shuffle(sub_non)
    sub_non = sub_non[ : 1000]
    print ("sub non: ", len(sub_non))
    newdata["testset"] = sub_non
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_sub_non.json", "w") as f:
        json.dump(newdata, f, indent=4)
    
    random.shuffle(con_ent)
    con_ent = con_ent[ : 1000]
    print ("con ent: ", len(con_ent))
    newdata["testset"] = con_ent
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_con_ent.json", "w") as f:
        json.dump(newdata, f, indent=4)
    
    random.shuffle(con_non)
    con_non = con_non[ : 1000]
    print ("con non: ", len(con_non))
    newdata["testset"] = con_non
    with open("/home/sichenglei/PromptQA/testsets/Spurious/hans_con_non.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_paws():
    newdata = {}
    newdata["dataset"] = "paws"
    newdata["testset"] = []

    with open("/home/sichenglei/LM-BFF/data/all_data_final/QQP_PAWS/full/test.tsv", "r") as f:
        data = f.readlines()[ 1 : ]
    
    for line in data:
        line = line.strip().split("\t")
        neweg = {}
        neweg["id"] = line[0]
        neweg["question"] = line[1].strip().replace("\n", " ") + "\n" + line[2].strip().replace("\n", " ") + "\n" + "Are these two questions asking the same thing?"
        if line[-1].strip() == "1":
            neweg["answer"] = "yes"
        else:
            neweg["answer"] = "no"
        newdata["testset"].append(neweg)

    print ("#test: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/Spurious/paws.json", "w") as f:
        json.dump(newdata, f, indent=4)

def sample_winobias(subset="pro_stereotyped_type1"):
    def extract(sent):
        ## helper function used to extract the reference and pronouns
        ## returns: [reference, pronoun]
        return re.findall(r'\[(.*?)\]', sent)

    with open(os.path.join("/home/sichenglei/QAdatasets/corefBias-master/WinoBias/wino/data", subset + ".txt.test"), "r") as f:
        data = f.readlines()
    
    newdata = {}
    newdata["dataset"] = "winobias"
    newdata["demos"] = []
    newdata["testset"] = []

    demos = data[ : 16]
    data = data[16 : ]

    for line in demos:
        sent = ' '.join(line.strip().split()[1 : ])
        entity = extract(sent)
        sent = sent.replace('[', '').replace(']', '') 
        
        neweg = {}
        neweg["question"] = sent + "\n" + "Who does '{}' refer to in the above sentence?".format(entity[1])
        neweg["answer"] = entity[0]

        newdata["demos"].append(neweg)
    
    for line in data:
        sent = ' '.join(line.strip().split()[1 : ])
        entity = extract(sent)
        sent = sent.replace('[', '').replace(']', '')
        
        neweg = {}
        neweg["question"] = sent + "\n" + "Who does '{}' refer to in the above sentence?".format(entity[1])
        neweg["answer"] = entity[0]

        newdata["testset"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))

    with open("/home/sichenglei/PromptQA/testsets/WinoBias/"+subset+".json", "w") as f:
        json.dump(newdata, f, indent=4)

def sample_winobias_prompt():
    demos = []
    for fp in ["/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type1.json", 
    "/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type2.json",
    "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type1.json",
    "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type2.json"]:
        with open(fp, "r") as f:
            data = json.load(f)
        demos.extend(data["demos"][ : 8])
    random.shuffle(demos)
    newdata = {}
    newdata["dataset"] = "winobias"
    newdata["demos"] = demos
    print ("#demos: ", len(newdata["demos"]))
    with open("/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_32shots.json", "w+") as f:
        json.dump(newdata, f, indent=4)


def sample_winobias_prompt_balanced_pro_at_end():
    demos = []
    anti = []
    pro = []
    for fp in ["/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type1.json", 
    "/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type2.json"]:
        with open(fp, "r") as f:
            data = json.load(f)
        anti.extend(data["demos"][ : 4])
    random.shuffle(anti)
    for fp in ["/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type1.json",
    "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type2.json"]:
        with open(fp, "r") as f:
            data = json.load(f)
        pro.extend(data["demos"][ : 4])
    random.shuffle(pro)
    newdata = {}
    newdata["dataset"] = "winobias"
    newdata["demos"] = anti + pro
    with open("/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_pro_at_end.json", "w+") as f:
        json.dump(newdata, f, indent=4)



def sample_winobias_prompt_balanced_anti_at_end():
    demos = []
    anti = []
    pro = []
    for fp in ["/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type1.json", 
    "/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type2.json"]:
        with open(fp, "r") as f:
            data = json.load(f)
        anti.extend(data["demos"][ : 4])
    random.shuffle(anti)
    for fp in ["/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type1.json",
    "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type2.json"]:
        with open(fp, "r") as f:
            data = json.load(f)
        pro.extend(data["demos"][ : 4])
    random.shuffle(pro)
    newdata = {}
    newdata["dataset"] = "winobias"
    newdata["demos"] = pro + anti
    with open("/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_anti_at_end.json", "w+") as f:
        json.dump(newdata, f, indent=4)


def sample_bbq():
    f_lst = os.listdir("/home/sichenglei/QAdatasets/BBQ/data")
    label_map = {0: 'A', 1: 'B', 2: 'C'}
    # print (f_lst)
    demos = []
    testset = []
    for ff in f_lst:
        with open(os.path.join("/home/sichenglei/QAdatasets/BBQ/data", ff), "r") as f:
            lines = f.readlines()
            for line in lines[:4]:
                demos.append(json.loads(line))
            for line in lines[4:1004]:
                testset.append(json.loads(line))
    print (len(demos), len(testset))

    newdata = {}
    newdata["dataset"] = "bbq"
    newdata["demos"] = []
    newdata["testset"] = []
    for line in demos:
        neweg = {}
        neweg["id"] = line["category"] + '-' + line["context_condition"] + '-' + str(line["question_index"]) + '-' + str(line["example_id"])
        neweg["question"] = line["context"] + "\n" + line["question"] + "\n"
        neweg["question"] += "A. " + line["ans0"] + "  B. " + line["ans1"] + "  C. " + line["ans2"]
        neweg["answer"] =  label_map[line["label"]]
        neweg["question_polarity"] = line["question_polarity"]
        neweg["answer_info"] = line["answer_info"]
        neweg["additional_metadata"] = line["additional_metadata"]
        newdata["demos"].append(neweg)
    
    for line in testset:
        neweg = {}
        neweg["id"] = line["category"] + '-' + line["context_condition"] + '-' + str(line["question_index"]) + '-' + str(line["example_id"])
        neweg["question"] = line["context"] + "\n" + line["question"] + "\n"
        neweg["question"] += "A. " + line["ans0"] + "  B. " + line["ans1"] + "  C. " + line["ans2"]
        neweg["answer"] =  label_map[line["label"]]
        neweg["question_polarity"] = line["question_polarity"]
        neweg["answer_info"] = line["answer_info"]
        neweg["additional_metadata"] = line["additional_metadata"]
        newdata["testset"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/BBQ/bbq.json", "w") as f:
        json.dump(newdata, f, indent=4)

def sample_edit_fever():
    label_mapping = {"SUPPORTS": "True", "REFUTES": "False"}
    data = []
    with open("/home/sichenglei/QAdatasets/datasets/fever-dev-kilt.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    newdata = {}
    newdata["dataset"] = "kilt"
    newdata["demos"] = []
    newdata["testset"] = []

    demos = data[ : 16]
    testset = data[16 : ]
    for i in range(len(demos)):
        neweg = {}
        neweg["id"] = demos[i]["id"]
        neweg["question"] = demos[i]["input"] + " True or False?"
        label = label_mapping[demos[i]["output"][0]["answer"]]
        neweg["answer"] = label 
        neweg["alternative"] = label_mapping[demos[i]["alternatives"][0]]
        neweg["rephrases"] = demos[i]["filtered_rephrases"]
        newdata["demos"].append(neweg)
    
    for i in range(len(testset)):
        neweg = {}
        neweg["id"] = testset[i]["id"]
        neweg["question"] = testset[i]["input"] + " True or False?"
        label = label_mapping[testset[i]["output"][0]["answer"]]
        neweg["answer"] = label 
        neweg["alternative"] = label_mapping[testset[i]["alternatives"][0]]
        neweg["rephrases"] = testset[i]["filtered_rephrases"]
        newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_orig_test_full.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_edit_qa():
    data = []
    with open("/home/sichenglei/QAdatasets/datasets/structured_zeroshot-dev-new_annotated_final.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    newdata = {}
    newdata["dataset"] = "zsre"
    newdata["demos"] = []
    newdata["testset"] = []

    demos = data[ : 16]
    testset = data[16 : ]
    for i in range(len(demos)):
        neweg = {}
        neweg["id"] = demos[i]["id"]
        neweg["question"] = demos[i]["input"]
        label = demos[i]["output"][0]["answer"]
        neweg["answer"] = label 
        neweg["alternative"] = demos[i]["alternatives"][0]
        neweg["rephrases"] = demos[i]["filtered_rephrases"]
        newdata["demos"].append(neweg)
    
    for i in range(len(testset)):
        neweg = {}
        neweg["id"] = testset[i]["id"]
        neweg["question"] = testset[i]["input"]
        label = testset[i]["output"][0]["answer"]
        neweg["answer"] = label 
        if len(testset[i]["alternatives"]) == 0:
            continue
        neweg["alternative"] = testset[i]["alternatives"][0]
        neweg["rephrases"] = testset[i]["filtered_rephrases"]
        newdata["testset"].append(neweg)

    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/zsre_orig_test_full.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_contriever_nq():
    from transformers import GPT2TokenizerFast
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data = []
    with open("/data3/private/clsi/Contriever/contriever/contriever_squad/test.jsonl", "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    
    with open("/home/sichenglei/PromptQA/testsets/AllQA/squad.json", "r") as f:
        newdata = json.load(f)
    
    retrieved = 0 ## how many questions have answer-containing contexts
    for i in range(len(data)):
        has_answer = 0
        neweg = newdata["testset"][i]
        for j in range(10):
            neweg["question"] = data[i]["ctxs"][j]["text"].replace("\n", " ") + "\n" + neweg["question"]
            has_answer = max(has_answer, int(data[i]["ctxs"][j]["hasanswer"]))
        neweg["has_answer"] = has_answer
        newdata["testset"][i] = neweg
        if has_answer:
            retrieved += 1

    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    print ("recall: ", retrieved / len(newdata["testset"]) * 100)
    with open("/home/sichenglei/PromptQA/testsets/retrieval/contriever_top10_squad.json", "w") as f:
        json.dump(newdata, f, indent=4)

def sample_edit_fever_filter():
    newdata = {}
    newdata["dataset"] = "fever"
    newdata["demos"] = []
    newdata["testset"] = []

    ## only save examples where orig predictions are different from intended predictions
    ## add in the knowledge update (in the original phrasing), then randomly sample a rephrase as the new test question
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_orig_test_full.json", "r") as f:
        data = json.load(f)
    newdata["demos"] = data["demos"]
    # data = data["testset"]
    # # print (len(data))

    # preds = []
    # with open("/home/sichenglei/PromptQA/logs/KnowUpdate/fever_orig_full_code002_16shot.log", "r") as f:
    #     lines = f.readlines()
    #     for i,line in enumerate(lines):
    #         if "Gold answer: " in lines[i]:
    #             p = lines[i-2].strip().split()[-1].strip()
    #             preds.append(eval(p))
    # # print (len(preds))

    # assert len(data) == len(preds), "length mismatch"
    # for i in range(len(preds)):
    #     if str(preds[i]) != data[i]["alternative"]:
    #         neweg = {}
    #         neweg["question"] = "Update: It is " + data[i]["alternative"] + " that " + data[i]["question"].replace("True or False?", "\n")
    #         if len(data[i]["rephrases"]) > 0:
    #             neweg["question"] += data[i]["rephrases"][0] + " True or False?"
    #             neweg["answer"] = data[i]["alternative"]
    #             newdata["testset"].append(neweg)
    
    # print ("#demos: ", len(newdata["demos"]))
    # print ("#testset: ", len(newdata["testset"]))
    # with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_edited_test.json", "w") as f:
    #     json.dump(newdata, f, indent=4)

    label_mapping = {"SUPPORTS": "True", "REFUTES": "False"}
    data = []
    with open("/home/sichenglei/QAdatasets/datasets/fever-train-kilt.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    random.shuffle(data)

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_edited_test.json", "r") as f:
        edited_data = json.load(f)
    edited_data = edited_data["testset"]
    
    for i in range(7871):
        neweg = {}
        neweg["id"] = data[i]["id"]
        neweg["question"] = edited_data[i]["question"].split("\n")[0] + "\n"
        neweg["question"] += data[i]["input"] + " True or False?"
        label = label_mapping[data[i]["output"][0]["answer"]]
        neweg["answer"] = label 
        newdata["testset"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_edited_irrelevant.json", "w") as f:
        json.dump(newdata, f, indent=4)


def sample_edit_qa_filter():
    newdata = {}
    newdata["dataset"] = "zsre"
    newdata["demos"] = []
    newdata["testset"] = []

    ## only save examples where orig predictions are different from intended predictions
    ## add in the knowledge update (in the original phrasing), then randomly sample a rephrase as the new test question
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdate/zsre_orig_test_full.json", "r") as f:
        data = json.load(f)
    newdata["demos"] = data["demos"]
    # data = data["testset"]
    # print (len(data))

    # preds = []
    # with open("/home/sichenglei/PromptQA/logs/KnowUpdate/zsre_orig_full_code002_16shot.log", "r") as f:
    #     lines = f.readlines()
    #     for i,line in enumerate(lines):
    #         if "Gold answer: " in lines[i]:
    #             p = lines[i-2].strip()[len("Answer:") : ].strip()
    #             preds.append(p)
    # # print (len(preds))

    # assert len(data) == len(preds), "length mismatch"
    # for i in range(len(preds)):
    #     ## sample a small set for eval
    #     if single_ans_f1(preds[i], data[i]["alternative"]) < 0.6 and i % 2 == 0 and len(data[i]["alternative"]) > 0:
    #         neweg = {}
    #         neweg["question"] = "Update: The answer to the question " + data[i]["question"].replace("?", "") + " is " + data[i]["alternative"] + ".\n"
    #         if len(data[i]["rephrases"]) > 0:
    #             neweg["question"] += data[i]["rephrases"][0]
    #             neweg["answer"] = data[i]["alternative"]
    #             newdata["testset"].append(neweg)
    
    # print ("#demos: ", len(newdata["demos"]))
    # print ("#testset: ", len(newdata["testset"]))
    # with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_test.json", "w") as f:
    #     json.dump(newdata, f, indent=4)

    data = []
    with open("/home/sichenglei/QAdatasets/datasets/structured_zeroshot-train-new_annotated_final.jsonl", "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    random.shuffle(data)

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_test.json", "r") as f:
        edited_data = json.load(f)
    edited_data = edited_data["testset"]
    
    for i in range(11916):
        neweg = {}
        neweg["id"] = data[i]["id"]
        # neweg["question"] = edited_data[i]["question"].split("\n")[0] + "\n"
        neweg["question"] = data[i]["input"]
        label = data[i]["output"][0]["answer"]
        neweg["answer"] = label 
        newdata["testset"].append(neweg)
    
    print ("#demos: ", len(newdata["demos"]))
    print ("#testset: ", len(newdata["testset"]))
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_orig_irrelevant.json", "w") as f:
        json.dump(newdata, f, indent=4)

def sample_edit_fever_balanced():
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/fever_edited_test.json", "r") as f:
        data = json.load(f)
    demos = data["demos"][ : 8] + data["testset"][ : 8]
    random.shuffle(demos)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_test.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/fever_orig_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_orig_irrelevant.json", "w") as f:
        json.dump(data, f, indent=4)
    

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/fever_edited_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_irrelevant.json", "w") as f:
        json.dump(data, f, indent=4)


def sample_edit_zsre_balanced():
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_test.json", "r") as f:
        data = json.load(f)
    demos = data["demos"][ : 8] + data["testset"][ : 8]
    random.shuffle(demos)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_edited_test.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_orig_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_orig_irrelevant.json", "w") as f:
        json.dump(data, f, indent=4)
    

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_edited_irrelevant.json", "w") as f:
        json.dump(data, f, indent=4)


def sample_edit_fever_balanced_all():
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_irrelevant.json", "r") as f:
        data = json.load(f)
    demos = data["demos"] + data["testset"][ : 8]
    random.shuffle(demos)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_edited_irrelevant_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_test.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_edited_test_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)
    

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_orig_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_orig_irrelevant_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)


def sample_edit_zsre_balanced_all():
    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_edited_irrelevant.json", "r") as f:
        data = json.load(f)
    demos = data["demos"] + data["testset"][ : 8]
    random.shuffle(demos)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_edited_irrelevant_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_edited_test.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_edited_test_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)
    

    with open("/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_orig_irrelevant.json", "r") as f:
        data = json.load(f)
    data["demos"] = demos 
    data["testset"] = data["testset"][ 8 : ]

    with open("/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_orig_irrelevant_all_balance.json", "w") as f:
        json.dump(data, f, indent=4)


def sample_mmlu():
    # for subset in ["astronomy", "us_foreign_policy", "nutrition"]:
    demos = []
    testset = []
    options = ['(A) ', '(B) ', '(C) ', '(D) ']
    for dirpath,_,filenames in os.walk("/data2/private/clsi/QAdatasets/MMLU/dev"):
        for f in filenames:
            fc = os.path.abspath(os.path.join(dirpath, f))
            set_name = f[ : -len("_dev.csv")]
            subcat = subcategories[set_name][0]
            maincat = None
            for k,v in categories.items():
                if subcat in v:
                    maincat = k 
            
            with open(fc) as f:
                reader = csv.reader(f, delimiter=',')
                for idx, row in enumerate(reader):
                    eg = {}
                    eg["id"] = str(idx)
                    eg["question"] = row[0].strip() + '\n'
                    for i in range(4):
                        eg["question"] += options[i] + row[i+1].strip()
                        if i < 3:
                            eg["question"] += '\n'
                    eg["answer"] = row[-1].strip()
                    eg["set_name"] = set_name
                    eg["subcat"] = subcat 
                    eg["maincat"] = maincat
                    demos.append(eg)
        
        # print (demos)
        # print (len(demos))

        for dirpath,_,filenames in os.walk("/data2/private/clsi/QAdatasets/MMLU/test"):
            for f in filenames:
                fc = os.path.abspath(os.path.join(dirpath, f))
                set_name = f[ : -len("_test.csv")]
                subcat = subcategories[set_name][0]
                maincat = None
                for k,v in categories.items():
                    if subcat in v:
                        maincat = k 
                
                with open(fc) as f:
                    reader = csv.reader(f, delimiter=',')
                    for idx, row in enumerate(reader):
                        eg = {}
                        eg["id"] = str(idx)
                        eg["question"] = row[0].strip() + '\n'
                        for i in range(4):
                            eg["question"] += options[i] + row[i+1].strip()
                            if i < 3:
                                eg["question"] += '\n'
                        eg["answer"] = row[-1].strip()
                        eg["set_name"] = set_name
                        eg["subcat"] = subcat 
                        eg["maincat"] = maincat
                        testset.append(eg)
        
        print ("#test: ", len(testset))
        
        # test_set = os.path.join("/data2/private/clsi/QAdatasets/MMLU/test", subset+"_test.csv") 
        # with open(test_set) as f:
        #     reader = csv.reader(f, delimiter=',')
        #     for idx, row in enumerate(reader):
        #         eg = {}
        #         eg["id"] = str(idx)
        #         eg["question"] = row[0].strip() + '\n'
        #         for i in range(4):
        #             eg["question"] += options[i] + row[i+1].strip()
        #             if i < 3:
        #                 eg["question"] += '\n'
        #         eg["answer"] = row[-1].strip()
        #         testset.append(eg)

        # print ("subset: ", subset)
        
        print ("#demos: ", len(demos))
        print ("#test: ", len(testset))
        newdata = {}
        newdata["dataset"] = "MMLU-all"
        newdata["demos"] = demos 
        newdata["testset"] = testset 

        save_dir = os.path.join("/home/sichenglei/PromptQA/testsets/MMLU/MMLU_all.json")
        with open(save_dir, "w+") as f:
            json.dump(newdata, f, indent=4)
        

## re-format MMLU to NQ format to feed into retriever 
def reformat_mmlu():
    options = ['(A) ', '(B) ', '(C) ', '(D) ']
    new_dev_data = []
    for dirpath,_,filenames in os.walk("/data2/private/clsi/QAdatasets/MMLU/dev"):
        for f in filenames:
            fc = os.path.abspath(os.path.join(dirpath, f))
            set_name = f[ : -len("_dev.csv")]
            subcat = subcategories[set_name][0]
            maincat = None
            for k,v in categories.items():
                if subcat in v:
                    maincat = k 
            
            with open(fc) as f:
                reader = csv.reader(f, delimiter=',')
                for idx, row in enumerate(reader):
                    eg = {}
                    eg["id"] = str(idx)
                    eg["question"] = row[0].strip() + '\n'
                    for i in range(4):
                        eg["question"] += options[i] + row[i+1].strip()
                        if i < 3:
                            eg["question"] += '\n'
                    eg["answer"] = [row[-1].strip()]
                    eg["set_name"] = set_name
                    eg["subcat"] = subcat 
                    eg["maincat"] = maincat
                    new_dev_data.append(eg)
        print ("#dev: ", len(new_dev_data))
        save_dir = os.path.join("/data3/private/clsi/qa_data/MMLU/MMLU_all_dev.json")
        with open(save_dir, "w+") as f:
            json.dump(new_dev_data, f, indent=4)
        
       
        new_test_data = []
        for dirpath,_,filenames in os.walk("/data2/private/clsi/QAdatasets/MMLU/test"):
            for f in filenames:
                fc = os.path.abspath(os.path.join(dirpath, f))
                set_name = f[ : -len("_test.csv")]
                subcat = subcategories[set_name][0]
                maincat = None
                for k,v in categories.items():
                    if subcat in v:
                        maincat = k 
                
                with open(fc) as f:
                    reader = csv.reader(f, delimiter=',')
                    for idx, row in enumerate(reader):
                        eg = {}
                        eg["id"] = str(idx)
                        eg["question"] = row[0].strip() + '\n'
                        for i in range(4):
                            eg["question"] += options[i] + row[i+1].strip()
                            if i < 3:
                                eg["question"] += '\n'
                        eg["answer"] = [row[-1].strip()]
                        eg["set_name"] = set_name
                        eg["subcat"] = subcat 
                        eg["maincat"] = maincat
                        new_test_data.append(eg)
        
        print ("#test: ", len(new_test_data))
        save_dir = os.path.join("/data3/private/clsi/qa_data/MMLU/MMLU_all_test.json")
        with open(save_dir, "w+") as f:
            json.dump(new_test_data, f, indent=4)
        

def sample_contriever_mmlu():
    from transformers import GPT2TokenizerFast
    gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # for subset in ["astronomy", "us_foreign_policy", "nutrition"]:
    dev_data = []
    dev_set = os.path.join("/data3/private/clsi/Contriever/contriever/MMLU/MMLU_all_dev.json") 
    with open(dev_set, "r") as f:
        lines = f.readlines()
        for line in lines:
            dev_data.append(json.loads(line))
    
    test_data = []
    test_set = os.path.join("/data3/private/clsi/Contriever/contriever/MMLU/MMLU_all_test.json") 
    with open(test_set, "r") as f:
        lines = f.readlines()
        for line in lines:
            test_data.append(json.loads(line))
    
    orig_data = os.path.join("/home/sichenglei/PromptQA/testsets/MMLU/MMLU_all.json")
    with open(orig_data, "r") as f:
        newdata = json.load(f)
    assert len(dev_data) == len(newdata["demos"]), "dev set length mismatch"
    assert len(test_data) == len(newdata["testset"]), "test set length mismatch"
    
    total_len = 0
    max_len = 0
    for i in range(len(dev_data)):
        neweg = newdata["demos"][i]
        for j in range(5):
            neweg["question"] = dev_data[i]["ctxs"][j]["text"].replace("\n", " ") + "\n" + neweg["question"]
        qlen = len(gpt_tokenizer.tokenize(neweg["question"]))
        max_len = max(max_len, qlen)
        total_len += qlen 
        newdata["demos"][i] = neweg

    print ("#demos: ", len(newdata["demos"]))
    print ("question max len: ", max_len)
    print ("question avg len: ", total_len / len(newdata["demos"]))
    

    total_len = 0
    max_len = 0
    for i in range(len(test_data)):
        neweg = newdata["testset"][i]
        for j in range(10):
            neweg["question"] = test_data[i]["ctxs"][j]["text"].replace("\n", " ") + "\n" + neweg["question"]
        qlen = len(gpt_tokenizer.tokenize(neweg["question"]))
        max_len = max(max_len, qlen)
        total_len += qlen 
        newdata["testset"][i] = neweg

    print ("#testset: ", len(newdata["testset"]))
    print ("question max len: ", max_len)
    print ("question avg len: ", total_len / len(newdata["testset"]))

    save_dir = os.path.join("/home/sichenglei/PromptQA/testsets/MMLU/MMLU_all_contriever.json")
    with open(save_dir, "w+") as f:
        json.dump(newdata, f, indent=4)




def lex_overlap(eg):
    all_words = 0

    prem_words = []
    hypo_words = []

    text_a = eg.split("\n")[0]
    text_b = eg.split("\n")[1]

    for p in puncs:
        prem = text_a.replace(p, ' ')
        hypo = text_b.replace(p, ' ')

    for word in prem.split():
        prem_words.append(word.lower())
    prem_words = list(set(prem_words))
    
    for word in hypo.split():
        hypo_words.append(word.lower())
    hypo_words = list(set(hypo_words))

    all_in = True
    overlap = 0
    for word in hypo_words:
        all_words += 1
        if word not in prem_words:
            all_in = False
            break
        else:
            overlap += 1
    
    return all_in 

def check_mnli_bias():
    with open("/home/sichenglei/PromptQA/testsets/GLUE/mnli_matched.json", "r") as f:
        data = json.load(f)
    
    demos = data["demos"][ : 48]
    # print (demos)
    ent = []
    non = []
    for di in range(len(demos)):
        if "entailment" not in demos[di]["answer"]:
            demos[di]["answer"] = "non-entailment"
            non.append(demos[di])
        else:
            demos[di]["answer"] = "entailment"
            ent.append(demos[di])
    
    with open("/home/sichenglei/LM-BFF/ConceptBenchFinal/MNLI/pos-support/sentence_pair/lexical_overlap_mnli/test_bias_support.tsv", "r") as f:
        biased = f.readlines()
    
    for i, line in enumerate(biased[ : 16]):
        question = line.split("\t")[0].strip() + "\n" + line.split("\t")[1].strip() + "\n" + "What is the relationship between these two sentences?"
        ent[i]["question"] = question

    demos = ent[: 16] + non[: 16]
    random.shuffle(demos)
    newdata = {}
    newdata["dataset"] = "MNLI-full-biased"
    newdata["demos"] = demos

    with open("/home/sichenglei/PromptQA/testsets/Spurious/MNLI_full_bias.json", "w+") as f:
        json.dump(newdata, f, indent=4)

if __name__ == '__main__':
    # sample("nq")
    # sample("triviaqa")
    # sample("squad")
    # sample("hotpotqa")
    # sample("webq")
    # sample("cwq")
    # sample("qampari")
    # sample("boolq")
    # sample("webqsp")
    # sample_strategyqa()
    # sample("timeqa-easy")
    # sample("timeqa-hard")
    # sample("timeqa-human-easy")
    # sample("timeqa-human-hard")
    # sample("grailqa")
    # sample("cfq-mcd1")
    # sample("cfq-mcd2")
    # sample("cfq-mcd3")
    # sample("freebaseqa")
    # sample("hybridqa")
    # sample("ambigqa")
    # sample("reclor")
    # sample("race")
    # sample("boolq-rc")
    # sample_nq_dpr()
    # sample("subqa-overall")
    # sample("subqa-sub1")
    # sample("subqa-sub2")
    # sample("subqa-overall-passage")
    # sample("subqa-all10")
    # sample("mrqa-nq-train")
    # sample("mrqa-squad-train")
    # sample("mrqa-newsqa-train")
    # sample_glue("mnli")
    # sample_glue("mnli-mm")
    # sample_glue("qnli")
    # sample_glue("qqp")
    # sample_glue("rte")
    # sample_glue("sst2")
    # sample_adv_glue("sst2")
    # sample_adv_glue("rte")
    # sample_adv_glue("qnli")
    # sample_adv_glue("mnli-mm")
    # sample_adv_glue("qqp")
    # sample_contrast_imdb()
    # sample_contrast_boolq()
    # sample_contrast_quoref()
    # sample_contrast_mctaco()
    # sample_squad_ood()
    # sample_glue("snli")
    # sample_nli("wnli")
    # sample_hans()
    # sample_paws()
    # sample_winobias("anti_stereotyped_type2")
    # sample_winobias_prompt()
    # sample_bbq()
    # sample_edit_qa()
    # sample_contriever_nq()
    # sample_edit_fever_filter()
    # sample_edit_qa_filter()
    # sample_edit_fever_balanced()
    # sample_edit_zsre_balanced()
    # sample_edit_fever_balanced_all()
    # sample_edit_zsre_balanced_all()
    # sample_winobias_prompt_balanced_pro_at_end()
    # sample_winobias_prompt_balanced_anti_at_end()
    # sample_winobias_prompt()
    # sample_mmlu()
    # reformat_mmlu()
    sample_contriever_mmlu()
    # reformat_mmlu()
    # check_mnli_bias()
    # check_mnli_bias()

    # sample_train("nq")
    # sample_train("hotpotqa")
    # sample_train("webqsp")
    # sample_train("nq")
    
    # with open(train_mappings["ambigqa"]) as f:
    #     data = json.load(f)
    # for line in data:
    #     if len(line["annotations"]) != 1:
    #         print (len(line["annotations"]))
    #         print (line)
    #         print ()
    
    # data = []
    # with open(train_mappings["qampari"]) as f:
    #     for line in f:
    #         line = json.loads(line)
    #         # print (line.keys())
    #         # print (line["question_text"])
    #         # print (line["answer_list"])
    #         # print (line["answer_list"][0].keys())
    #         print (line["answer_list"][0]["aliases"])
    #         # if len(data) >= 5:
    #         break

    # with open(test_mappings["ambigqa"]) as f:
    #     data = json.load(f)
    # # print (data[0].keys())
    # print (data[10]["annotations"])


    # data = []
    # with open("/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/test.easy.json", "r") as f:
    
    # with open("/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_test.hard.json", "r") as f:
    #     for line in f:
    #         data.append(json.loads(line))
    # print (len(data))
    # noans = 0
    # for line in data:
    #     if line['targets'][0] == '':
    #         noans += 1
    # print (noans)

    # print (data[11]['question'])
    # print (data[11]['targets'])

    # easy = []
    # with open("/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_test.easy.json", "r") as f:
    #     for line in f:
    #         easy.append(json.loads(line))
    
    # hard = []
    # with open("/home/sichenglei/QAdatasets/Time-Sensitive-QA/dataset/human_test.hard.json", "r") as f:
    #     for line in f:
    #         hard.append(json.loads(line))

    # counter = 0
    # for i in range(len(easy)):
    #     if easy[i]["question"] == hard[i]["question"]:
    #         counter += 1
    #         print (easy[i]["question"], hard[i]["question"])
    # print (counter)

    # with open("/home/sichenglei/QAdatasets/GrailQA_v1.0/grailqa_v1.0_dev.json", "r") as f:
    #     data = json.load(f)
    # levels = []
    # for d in data:
    #     levels.append(d["level"])
    # print (set(levels))


    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd1_test.json", "r") as f:
    #     data = json.load(f)
    # print (len(data))
    # print (data[20]["question"])    
    # print (data[20]["expectedResponse"])    
    # print (data[200]["question"])    
    # print (data[200]["expectedResponse"]) 
    

    # with open("/home/sichenglei/QAdatasets/cfq/dataset.json", "r") as f:
    #     data = json.load(f)
    # mcd1_train = []
    # mcd1_test = []
    # mcd2_train = []
    # mcd2_test = []
    # mcd3_train = []
    # mcd3_test = []

    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd1.json", 'r') as f:
    #     mcd1 = json.load(f)
    # random.shuffle(mcd1["trainIdxs"])
    # for idx in mcd1["trainIdxs"][ : 1024]:
    #     mcd1_train.append(data[idx])
    # for idx in mcd1["testIdxs"]:
    #     mcd1_test.append(data[idx])

    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd1_train.json", "w") as f:
    #     json.dump(mcd1_train, f, indent=4)
    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd1_test.json", "w") as f:
    #     json.dump(mcd1_test, f, indent=4)



    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd2.json", 'r') as f:
    #     mcd2 = json.load(f)
    # random.shuffle(mcd2["trainIdxs"])
    # for idx in mcd2["trainIdxs"][ : 1024]:
    #     mcd2_train.append(data[idx])
    # for idx in mcd2["testIdxs"]:
    #     mcd2_test.append(data[idx])

    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd2_train.json", "w") as f:
    #     json.dump(mcd2_train, f, indent=4)
    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd2_test.json", "w") as f:
    #     json.dump(mcd2_test, f, indent=4)

    

    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd3.json", 'r') as f:
    #     mcd3 = json.load(f)
    # random.shuffle(mcd3["trainIdxs"])
    # for idx in mcd3["trainIdxs"][ : 1024]:
    #     mcd3_train.append(data[idx])
    # for idx in mcd3["testIdxs"]:
    #     mcd3_test.append(data[idx])

    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd3_train.json", "w") as f:
    #     json.dump(mcd3_train, f, indent=4)
    # with open("/home/sichenglei/QAdatasets/cfq/splits/mcd3_test.json", "w") as f:
    #     json.dump(mcd3_test, f, indent=4)


    # with open("/home/sichenglei/QAdatasets/FreebaseQA/FreebaseQA-eval.json", "r") as f:
    #     data = json.load(f)
    # data = data["Questions"]
    # print (len(data))
    # print (data[0].keys())
    # print (data[0]['Parses'])
 
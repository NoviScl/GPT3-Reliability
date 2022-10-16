import os
import csv
import argparse
import numpy as np
import multiprocessing
import time
import json
import time
from time import sleep
import random
import openai
from tqdm import tqdm
from utils import *

task_mappings = {
    "hotpotqa": "/home/sichenglei/NatIns/tasks/task1293_kilt_tasks_hotpotqa_question_answering.json",
    "multirc": "/home/sichenglei/NatIns/tasks/task058_multirc_question_answering.json",
    "aqua": "/home/sichenglei/NatIns/tasks/task750_aqua_multiple_choice_answering.json"
}

subset_mappings = {
    "aqua": "NI_subset/task750_aqua_1k.json",
    "gsm8k": "NI_subset/gsm8k_1k.json"
}

def PromptStep(args, prompt, temp):
    ## A Single Prompt Step
    response = openai.Completion.create(
        engine=args.engine,
        prompt=prompt,
        max_tokens=args.maxlen,
        logprobs=1,
        temperature=temp,
        stream=False,
        stop=["<|endoftext|>", "Question"]
    )
   
    output = response['choices'][0]["text"].strip()

    if args.extract:  
        prompt += output + "\n"
        prompt += "Therefore, the final answer is "

        response = openai.Completion.create(
            engine=args.engine,
            prompt=prompt,
            max_tokens=args.maxlen,
            logprobs=1,
            temperature=temp,
            stream=False,
            stop=["<|endoftext|>", "Question"]
        )
        
        output = response['choices'][0]["text"].strip()
    
    return output, prompt

def SelfConPrompt(args, counter, prompt, eg):
    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        print (prompt)
    
    all_outputs = []
    ## self-consistency prompting
    ## we sample 10 different outputs with temperature 0.7
    for i in range(10):
        try:
            output, newprompt = PromptStep(args, prompt, temp=0.7)
        except:
            sleep(10)
            output, newprompt = PromptStep(args, prompt, temp=0.7)

        if args.task == "aqua":
            ans = answer_extract_aqua(output)
        
        if args.print:
            print ("\nOutput #{}".format(str(i+1)))
            print (output)
            print ("\nExtracted answer string: ", ans)
        
        ## exclude no-answer cases
        if ans is not None:
            all_outputs.append(ans)
    
    final_ans = most_common(all_outputs)
    if args.task == "aqua":
        match = answer_match_aqua(final_ans, eg["output"][0])
    
    if args.print:
        print ("\n\nQuestion #{} Summary: ".format(str(counter)))
        print ("All predicted answers: ", all_outputs)
        print ("Final prediction: ", final_ans)
        print ("\nGold answer: ", eg["output"][0])
        print ("Correct? ", match)
        print ("\n\n")
    
    return match 


def SinglePrompt(args, counter, prompt, eg):
    ## greedy decoding by default
    try:
        output, newprompt = PromptStep(args, prompt, temp=0.)
    except:
        sleep(10)
        output, newprompt = PromptStep(args, prompt, temp=0.)

    if args.task == "aqua":
        match = answer_match_aqua(output, eg["output"][0])

    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        print (newprompt + output)
        print ("\nGold answer: ", eg["output"][0])
        print ("Correct? ", match)
        print ("\n\n")
    
    return match


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--task', type=str, help='specify the task that you want to evaluate')
    parser.add_argument('--prompt', type=str, default=None, help='specify the prompting method', choices=["zeroshot", "zeroshot-step", "fewshot", "fewshot-cot", "fewshot-cot-selfcon"])
    parser.add_argument('--print', default=False, action='store_true', help='Whether to print out every prompt')
    parser.add_argument('--extract', default=False, action='store_true', help='Whether to add an additional answer extraction step')
    parser.add_argument('--subset', default=False, action='store_true', help='Whether to use a small subset for debugging')
    parser.add_argument('--maxlen', type=int, default=256, help='max number of tokens to be generated')

    args = parser.parse_args()
    openai.api_key = args.apikey

    correct = 0
    counter = 801

    if args.task in subset_mappings:
        task_dir = subset_mappings[args.task]
        with open(task_dir, "r") as f:
            data = json.load(f)
        instruction = data["Definition"][0]
        test_set = data["Instances"]

        if args.subset:
            test_set = test_set[801 : ]

        print ("Size of test set:", len(test_set))
    else:
        print ("Task is out of our data collection")
        return 

    for eg in tqdm(test_set):
        counter += 1

        prompt = ""
        prompt += instruction + "\n\n"
        
        ## few-shot demo
        if args.prompt in ["fewshot", "fewshot-cot", "fewshot-cot-selfcon"]:
            with open("cot_demos.json", "r") as f:
                cot_demos = json.load(f)
            cot_demos = cot_demos[args.task]
            # print ("Number of demo examples: ", len(cot_demos))
            for demo in cot_demos:
                prompt += demo["question"] + "\n"
                
                if args.prompt == "fewshot":
                    ## without cot
                    prompt += "Answer: " + demo["answer"] + "\n\n"
                else:
                    ## with cot
                    prompt += "Answer: Let’s think step by step. " + demo["cot"] + "\n"
                    prompt += "Therefore, the final answer is " + demo["answer"] + "\n\n"
        
        ## current test instance 
        prompt += eg["input"]  + "\n"
        prompt += "Answer: "

        if args.prompt in ["zeroshot-step", "fewshot-cot", "fewshot-cot-selfcon"]:
            prompt += "Let’s think step by step. "
        
        if args.prompt == "fewshot-cot-selfcon":
            match = SelfConPrompt(args, counter, prompt, eg)
        else:
            match = SinglePrompt(args, counter, prompt, eg)

        if match:
            correct += 1
    
    # print ("correct: ", correct)
    # print ("counter: ", counter)
    print ("Accuracy: {}/{}={}%".format(correct, counter, correct / counter * 100))
    

if __name__ == '__main__':
    main()
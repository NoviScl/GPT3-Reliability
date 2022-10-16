baseline = "/home/sichenglei/RealPrompt/logs/aqua_code002_fewshot.log"
prompt = "/home/sichenglei/RealPrompt/logs/aqua_code002_fewshot_cot.log"

baseline_correct = []
prompt_correct = []

with open(baseline, "r") as f:
    for line in f.readlines():
        if "Correct?  " in line:
            baseline_correct.append(line.strip())

with open(prompt, "r") as f:
    for line in f.readlines():
        if "Correct?  " in line:
            prompt_correct.append(line.strip())

assert len(baseline_correct) == 1000, "wrong length"
assert len(prompt_correct) == 1000, "wrong length"

wrong_to_correct = []
correct_to_wrong = []
wrong_to_wrong = []
for i in range(1000):
    if baseline_correct[i] == "Correct?  True" and prompt_correct[i] == "Correct?  False":
        correct_to_wrong.append(i + 1)
    if baseline_correct[i] == "Correct?  False" and prompt_correct[i] == "Correct?  True":
        wrong_to_correct.append(i + 1)
    if baseline_correct[i] == "Correct?  False" and prompt_correct[i] == "Correct?  False":
        wrong_to_wrong.append(i + 1)

print ("wrong_to_correct: ", len(wrong_to_correct))
print ("correct_to_wrong: ", len(correct_to_wrong))
print ("wrong_to_wrong: ", len(wrong_to_wrong))

print ("wrong_to_correct: ", wrong_to_correct)
print ("wrong_to_wrong: ", wrong_to_wrong)
## tmux 24
## AQUA
# fewshot-cot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_text001_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_text002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_code002_fewshot_cot.log

## GSM8K
## fewshot-cot
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt fewshot-cot --maxlen 256 > logs/gsm8k_text001_fewshot_cot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot-cot --maxlen 256 > logs/gsm8k_text002_fewshot_cot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt fewshot-cot --maxlen 256 > logs/gsm8k_code002_fewshot_cot.log

# GSM8K
## fewshot-cot-selfcon
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/gsm8k_text002_fewshot_cot_selfcon.log


## MultiArith
## fewshot-cot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_text001_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_text002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_code002_fewshot_cot.log

## SingleEq
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task singleeq --print --prompt fewshot-cot --maxlen 256 --extract > logs/singleeq_text001_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task singleeq --print --prompt fewshot-cot --maxlen 256 --extract > logs/singleeq_text002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task singleeq --print --prompt fewshot-cot --maxlen 256 --extract > logs/singleeq_code002_fewshot_cot.log


## HotpotQA
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_text001_fewshot_l2m.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_code002_fewshot_l2m.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_text002_fewshot_l2m.log


python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_ood_code002_fewshot_cot.log
python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_ood_text001_fewshot_cot.log
python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_ood_text002_fewshot_cot.log


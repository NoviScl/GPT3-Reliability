## tmux 23
### AQUA
# fewshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_text001_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_text002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_code002_fewshot.log


### GSM8K
## fewshot
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_text001_fewshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_text002_fewshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_code002_fewshot.log


## fewshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_text001_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_text002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_code002_fewshot.log

# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task singleeq --print --prompt fewshot --maxlen 64 --extract > logs/singleeq_text001_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task singleeq --print --prompt fewshot --maxlen 64 --extract > logs/singleeq_code002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task singleeq --print --prompt fewshot --maxlen 64 --extract > logs/singleeq_text002_fewshot.log

# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua-advanced --print --prompt fewshot --maxlen 64 --extract > logs/aqua_advanced_code002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua-advanced --print --prompt fewshot-cot --maxlen 256 --extract > logs/aqua_advanced_code002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua-advanced --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/aqua_advanced_code002_fewshot_cot_selfcon.log

# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task hotpotqa --print --prompt fewshot --maxlen 64 > logs/hotpotqa_ood_code002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task hotpotqa --print --prompt fewshot --maxlen 64 > logs/hotpotqa_ood_text001_fewshot.log
python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task hotpotqa --print --prompt fewshot --maxlen 64 > logs/hotpotqa_ood_text002_fewshot.log

# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task hotpotqa --print --prompt fewshot-cot --maxlen 256 > logs/hotpotqa_text002_fewshot_l2m.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task hotpotqa --print --prompt zeroshot-boost --maxlen 256 --extract > logs/hotpotqa_text001_zeroshot_boost_extract_new.log

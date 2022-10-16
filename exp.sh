#### AQUA
# ## zeroshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt zeroshot --maxlen 64 > logs/aqua_text001_zeroshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt zeroshot --maxlen 64 > logs/aqua_text002_zeroshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt zeroshot --maxlen 128 > logs/aqua_code002_zeroshot.log

# ## zeroshot-step
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_text001_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_text002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_code002_zeroshot_step.log

## fewshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_text001_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_text002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt fewshot --maxlen 64 > logs/aqua_code002_fewshot.log

## fewshot-cot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_text001_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_text002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt fewshot-cot --maxlen 256 > logs/aqua_code002_fewshot_cot.log

## fewshot-cot-selfcon
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/aqua_text001_fewshot_cot_selfcon.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/aqua_text002_fewshot_cot_selfcon.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/aqua_code002_fewshot_cot_selfcon.log


#### GSM8K
# ## zeroshot
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt zeroshot --maxlen 64 --extract > logs/gsm8k_text001_zeroshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt zeroshot --maxlen 64 --extract > logs/gsm8k_text002_zeroshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt zeroshot --maxlen 64 --extract > logs/gsm8k_code002_zeroshot.log

# ## zeroshot-step 
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt zeroshot-step --maxlen 256 --extract
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt zeroshot-step --maxlen 256 --extract
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt zeroshot-step --maxlen 256 --extract

## fewshot
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_text001_fewshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_text002_fewshot.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt fewshot --maxlen 64 --extract > logs/gsm8k_code002_fewshot.log

## fewshot-cot
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt fewshot-cot --maxlen 256 
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot-cot --maxlen 256 
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt fewshot-cot --maxlen 256 

## fewshot-cot-selfcon
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/gsm8k_text001_fewshot_cot_selfcon.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/gsm8k_text002_fewshot_cot_selfcon.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/gsm8k_code002_fewshot_cot_selfcon.log



# #### MultiArith
# # ## zeroshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_text001_zeroshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_text002_zeroshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_code002_zeroshot.log

# ## zeroshot-step 
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_text001_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_text002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_code002_zeroshot_step.log

## fewshot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_text001_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_text002_fewshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt fewshot --maxlen 64 --extract > logs/multiarith_code002_fewshot.log

## fewshot-cot
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_text001_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_text002_fewshot_cot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt fewshot-cot --maxlen 256 --extract > logs/multiarith_code002_fewshot_cot.log

## fewshot-cot-selfcon
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/multiarith_text001_fewshot_cot_selfcon.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/multiarith_text002_fewshot_cot_selfcon.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt fewshot-cot-selfcon --maxlen 256 > logs/multiarith_code002_fewshot_cot_selfcon.log

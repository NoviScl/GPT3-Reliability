## tmux 26
# fewshot-cot-selfcon
# python -u cot_play.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt fewshot-cot-selfcon --maxlen 256 --subset > logs/aqua_text002_fewshot_cot_selfcon_801onwards.log

python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task singleeq --print --prompt fewshot-cot-selfcon --maxlen 256 --extract > logs/singleeq_text001_fewshot_cot_selfcon.log

for dataset in bbq
do 
    python -u cot.py \
    --apikey  \
    --engine code-davinci-002 \
    --task $dataset \
    --prompt_source bbq_ambig_pro \
    --prompt_method fewshot \
    --print \
    --maxlen 2 \
    --shots 8 
done

# > logs/BBQ/${dataset}_ambig_pro_prompt_intervention_code002_8shot.log

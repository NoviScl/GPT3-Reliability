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

# for dataset in nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 2 > logs/calibration/${dataset}_probs_code002_2shot.log
# done

# for dataset in mmlu_all_closed_book
# do 
#     python -u cot_mmlu.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 1 \
#     --shots 5 > logs/MMLU/${dataset}_code002_5shot.log
# done


# for dataset in hans_sub_non hans_con_ent hans_con_non hans_lex_ent hans_lex_non hans_sub_ent 
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli_half_bias \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/SpuriousNew/mnli_half_bias_${dataset}_code002_32shot.log
# done


# for dataset in hans_sub_non hans_con_ent hans_con_non hans_lex_ent hans_lex_non hans_sub_ent 
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli_full_bias \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/SpuriousNew/mnli_full_bias_${dataset}_code002_32shot.log
# done




# for dataset in mnli-hans-format
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli_half_bias \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/SpuriousNew/mnli_half_bias_${dataset}_code002_32shot.log
# done



# for dataset in mnli-hans-format
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli_full_bias \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/SpuriousNew/mnli_full_bias_${dataset}_code002_32shot.log
# done



# for dataset in mmlu_astronomy mmlu_nutrition mmlu_policy
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine davinci \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 1 \
#     --shots 5 > logs/MMLU/${dataset}_davinci_legacy_5shot.log
# done



# for dataset in mmlu_astronomy_top10_qonly mmlu_nutrition_top10_qonly mmlu_policy_top10_qonly
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine davinci \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 1 \
#     --shots 5 > logs/MMLU/${dataset}_davinci_legacy_5shot.log
# done


# for dataset in mmlu_astronomy_top10_qonly mmlu_nutrition_top10_qonly mmlu_policy_top10_qonly
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 2 \
#     --shots 5 > logs/MMLU/${dataset}_code002_5shot.log
# done



# for dataset in zsre_orig_full
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done

# for dataset in contriever_top5_triviaqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done

# for dataset in contriever_top5_nq_pqa  contriever_top5_squad_pqa  contriever_top5_triviaqa_pqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 4 > logs_new/retrieval/${dataset}_code002_4shot.log
# done

# for dataset in nq hotpotqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/${dataset}_probs_code002_16shot.log
# done

# > logs_new/retrieval/${dataset}_code002_4shot.log



# for dataset in triviaqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot-selfcon \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/${dataset}_selfcon_code002_16shot.log
# done


## OOD
# for dataset in hotpotqa triviaqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source nq \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/nq_to_${dataset}_probs_code002_16shot.log
# done


# ## different backbones
# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine text-davinci-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/${dataset}_probs_textD001_16shot.log
# done


# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine text-curie-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/${dataset}_probs_textC001_16shot.log
# done


# ## different shots
# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 8 > logs/calibration/${dataset}_probs_code002_8shot.log
# done


# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 32 > logs/calibration/${dataset}_probs_code002_32shot.log
# done


# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 64 > logs/calibration/${dataset}_probs_code002_64shot.log
# done



# for dataset in triviaqa nq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 128 > logs/calibration/${dataset}_probs_code002_128shot.log
# done



# for dataset in contriever_top10_triviaqa
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done


# for dataset in fever_edited_test fever_orig_irrelevant fever_edited_irrelevant
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done



# for dataset in fever_edited_test_all_balance fever_orig_irrelevant_all_balance fever_edited_irrelevant_all_balance
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done



# for dataset in bbq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source bbq_ambig_neutral \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 8 > logs/BBQ/${dataset}_ambig_neutral_prompt_code002_8shot.log
# done


# for dataset in bbq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source bbq_ambig_pro \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 8 > logs/BBQ/${dataset}_ambig_pro_prompt_code002_8shot.log
# done



# for dataset in bbq
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source bbq_ambig_anti \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 8 > logs/BBQ/${dataset}_ambig_anti_prompt_code002_8shot.log
# done



# for dataset in winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_balanced_prompt \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_debiased_code002_16shot.log
# done



# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_balanced_prompt_pro_at_end \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_balanced_prompt_pro_at_end_code002_16shot.log
# done



# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_balanced_prompt_anti_at_end \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_balanced_prompt_anti_at_end_code002_16shot.log
# done


# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_anti_type2 \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_anti_type2_code002_16shot.log
# done




# for dataset in qqp
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/Spurious/${dataset}_code002_32shot.log
# done


# for dataset in hans_sub_non
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 0 > logs/Spurious/mnli_to_${dataset}_code002_0shot.log
# done



# for dataset in hans_sub_non hans_con_ent hans_con_non hans_lex_ent hans_lex_non hans_sub_ent 
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 48 > logs/SpuriousNew/mnli_to_${dataset}_code002_48shot.log
# done


# for dataset in hans_sub_ent hans_sub_non hans_con_ent hans_con_non hans_lex_ent hans_lex_non 
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source mnli \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 48 > logs/Spurious/mnli_to_${dataset}_code002_48shot.log
# done



# for dataset in squad_amazon
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 8 > logs/SQuAD_OOD/${dataset}_code002_8shot.log
# done


# for dataset in IIDPromptMRQASQuADDev
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 8 > logs/SQuAD_OOD/${dataset}_code002_8shot.log
# done



# for dataset in qnli qqp
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/GLUE/${dataset}_code002_32shot.log
# done



# for dataset in mnli-mm
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 8 \
#     --shots 48 > logs/GLUE/${dataset}_code002_48shot.log
# done



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task subqa-all10 \
# --prompt_source subqa-all10 \
# --prompt_method fewshot \
# --print \
# --maxlen 16 \
# --shots 16 > logs/subqa-all10_code002_16shot.log


# for dataset in subs-nq-PmQAm-codeD002 subs-nq-PQA-codeD002 
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 8 > logs/KnowledgeConflict/${dataset}_code002_8shot.log
# done


# for dataset in subs-nq-QA-codeD002 subs-nq-QAm-codeD002
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/KnowledgeConflict/${dataset}_code002_16shot.log
# done


# for dataset in MinPromptMRQASQuADDev
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 8 > logs/MRQAMinPrompt/${dataset}_code002_8shot.log
# done


# for dataset in mrqa-nq-train mrqa-squad-train
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine text-davinci-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/MRQA-train-orig/${dataset}_textDavinci001_16shot.log
# done


# for dataset in mrqa-nq-train mrqa-squad-train
# do 
#     python -u cot.py \
#     --apikey  \
#     --engine text-curie-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/MRQA-train-orig/${dataset}_textCurie001_16shot.log
# done

# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task subqa-sub1 \
# --prompt_source subqa-sub1 \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/subqa-sub1_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task subqa-sub2 \
# --prompt_source subqa-sub2 \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/subqa-sub2_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task hotpotqa-train \
# --prompt_source hotpotqa-train \
# --prompt_method fewshot \
# --print \
# --maxlen 64 \
# --shots 16 > logs/hotpotqa-train_code002_16shot.log


# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task ambigqa \
# --prompt_source ambigqa \
# --prompt_method fewshot \
# --print \
# --maxlen 64 \
# --shots 16 > logs/ambigqa_code002_16shot.log

# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task ambigqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 64 \
# --shots 16 > logs/hotpotqa_to_ambigqa_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task grailqa \
# --prompt_source grailqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/grailqa_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task cfq-mcd1 \
# --prompt_source cfq-mcd1 \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/cfq-mcd1_code002_16shot.log


# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task cfq-mcd2 \
# --prompt_source cfq-mcd2 \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/cfq-mcd2_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task cfq-mcd3 \
# --prompt_source cfq-mcd3 \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/cfq-mcd3_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task timeqa-human-hard \
# --prompt_source timeqa-human-hard \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/timeqa-human-hard_code002_16shot.log



# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task hotpotqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot-cot \
# --print \
# --maxlen 128 \
# --shots 16 > logs/hotpotqa_code002_16shot_cot.log


# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task triviaqa \
# --prompt_source triviaqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/triviaqa_code002_16shot.log


# python -u cot.py \
# --apikey  \
# --engine code-davinci-002 \
# --task webq \
# --prompt_source webq \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/webq_code002_16shot.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-002 \
# --task nq \
# --prompt_source nq \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/nq_text002_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-001 \
# --task triviaqa \
# --prompt_source triviaqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/triviaqa_text001_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-002 \
# --task triviaqa \
# --prompt_source triviaqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/triviaqa_text002_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-001 \
# --task hotpotqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/hotpotqa_text001_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-002 \
# --task hotpotqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/hotpotqa_text002_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-001 \
# --task squad \
# --prompt_source squad \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/squad_text001_64shot_1k.log


# python -u cot.py \
# --apikey  \
# --engine text-davinci-002 \
# --task squad \
# --prompt_source squad \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --subset \
# --subset_size 1000 \
# --shots 64 > logs/squad_text002_64shot_1k.log



## zeroshot
# python -u cot.py --apikey  --engine text-davinci-001 --task aqua --print --prompt zeroshot --maxlen 64 > logs/aqua_text001_zeroshot.log
# python -u cot.py --apikey  --engine text-davinci-002 --task aqua --print --prompt zeroshot --maxlen 64 > logs/aqua_text002_zeroshot.log
# python -u cot.py --apikey  --engine code-davinci-002 --task aqua --print --prompt zeroshot --maxlen 128 > logs/aqua_code002_zeroshot.log


#### GSM8K
# ## zeroshot
# python -u cot.py --apikey  --engine text-davinci-001 --task gsm8k --print --prompt zeroshot --maxlen 64 --extract > logs/gsm8k_text001_zeroshot.log
# python -u cot.py --apikey  --engine text-davinci-002 --task gsm8k --print --prompt zeroshot --maxlen 64 --extract > logs/gsm8k_text002_zeroshot.log
# python -u cot.py --apikey  --engine code-davinci-002 --task gsm8k --print --prompt zeroshot-boost --maxlen 256 --extract > logs/gsm8k_code002_zeroshot.log

#### MultiArith
# ## zeroshot
# python -u cot.py --apikey  --engine text-davinci-001 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_text001_zeroshot.log
# python -u cot.py --apikey  --engine text-davinci-002 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_text002_zeroshot.log
# python -u cot.py --apikey  --engine code-davinci-002 --task multiarith --print --prompt zeroshot --maxlen 64 --extract > logs/multiarith_code002_zeroshot.log


### SingleEq
# python -u cot.py --apikey  --engine text-davinci-001 --task singleeq --print --prompt zeroshot --maxlen 64 --extract > logs/singleeq_text001_zeroshot.log
# python -u cot.py --apikey  --engine text-davinci-002 --task singleeq --print --prompt zeroshot --maxlen 64 --extract > logs/singleeq_text002_zeroshot.log
# python -u cot.py --apikey  --engine code-davinci-002 --task singleeq --print --prompt zeroshot --maxlen 64 --extract > logs/singleeq_code002_zeroshot.log

# python -u cot.py --apikey  --engine text-davinci-002 --task singleeq --print --prompt zeroshot-step --maxlen 256 --extract > logs/singleeq_text002_zeroshot_step.log
# python -u cot.py --apikey  --engine text-davinci-002 --task singleeq --print --prompt fewshot --maxlen 64 --extract > logs/singleeq_text002_fewshot.log

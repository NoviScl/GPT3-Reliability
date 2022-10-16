## tmux 22


for dataset in mmlu_all_contriever
do 
    python -u cot_mmlu.py \
    --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
    --engine code-davinci-002 \
    --task $dataset \
    --prompt_source mmlu_all_closed_book \
    --prompt_method fewshot \
    --print \
    --save_prob \
    --maxlen 1 \
    --shots 5 > logs/MMLU/${dataset}_prompt_closed_book_code002_5shot.log
done


for dataset in mmlu_all_contriever
do 
    python -u cot_mmlu.py \
    --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
    --engine code-davinci-002 \
    --task $dataset \
    --prompt_source mmlu_all_contriever \
    --prompt_method fewshot \
    --print \
    --save_prob \
    --maxlen 1 \
    --shots 5 > logs/MMLU/${dataset}_prompt_contriever_5shot.log
done


# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_balanced_prompt_32shots \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 32 > logs/WinoBias/${dataset}_balanced_prompt_32shots_code002.log
# done



# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_pro_type1_32shots \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 32 > logs/WinoBias/${dataset}_pro_type1_32shots_code002.log
# done


# for dataset in winobias_anti_type1 winobias_pro_type1
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_balanced_prompt \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_debiased_code002_16shot.log
# done

# for dataset in contriever_top5_nq
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done


# for dataset in contriever_top10_nq
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done


# for dataset in contriever_top5_squad
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done


# for dataset in contriever_top10_squad
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/retrieval/${dataset}_code002_16shot.log
# done


# for dataset in zsre_edited_irrelevant
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done


# for dataset in zsre_edited_test_balance zsre_orig_irrelevant_balance zsre_edited_irrelevant_balance
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done


# for dataset in zsre_edited_test_all_balance zsre_orig_irrelevant_all_balance zsre_edited_irrelevant_all_balance
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/KnowUpdate/${dataset}_code002_16shot.log
# done



# for dataset in nq hotpotqa
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot-selfcon \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/${dataset}_selfcon_code002_16shot.log
# done

# # > logs/calibration/${dataset}_probs_code002_16shot.log


# for dataset in triviaqa
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source hotpotqa \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/hotpotqa_to_${dataset}_probs_code002_16shot.log
# done




# for dataset in hotpotqa triviaqa
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source nq \
#     --prompt_method fewshot \
#     --print \
#     --save_prob \
#     --maxlen 32 \
#     --shots 16 > logs/calibration/nq_to_${dataset}_probs_code002_16shot.log
# done



# for dataset in bbq
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source bbq_disambig \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 8 > logs/BBQ/${dataset}_disambig_prompt_code002_8shot.log
# done


# for dataset in bbq
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source race \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 2 \
#     --shots 8 > logs/BBQ/${dataset}_race_prompt_code002_8shot.log
# done


# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_pro_type1 \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_pro_type1_code002_16shot.log
# done


# for dataset in winobias_anti_type1 winobias_pro_type1 winobias_anti_type2 winobias_pro_type2
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source winobias_pro_type2 \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/WinoBias/${dataset}_pro_type2_code002_16shot.log
# done





# for dataset in paws
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source qqp \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/SpuriousNew/qqp_to_${dataset}_code002_32shot.log
# done


# for dataset in mnli
# do 
#     python -u cot2.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 48 > logs/Spurious/${dataset}_code002_48shot.log
# done


# for dataset in qqp
# do 
#     python -u cot2.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/Spurious/${dataset}_code002_32shot.log
# done


# for dataset in snli
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 48 > logs/GLUE/${dataset}_code002_48shot.log
# done


# for dataset in squad_nyt squad_reddit 
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 8 > logs/SQuAD_OOD/${dataset}_code002_8shot.log
# done

# for dataset in mnli-mm
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 8 \
#     --shots 48 > logs/GLUE/${dataset}_code002_48shot.log
# done


# for dataset in adv-qqp
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 4 \
#     --shots 32 > logs/AdvGLUE/${dataset}_code002_32shot.log
# done

# for dataset in subs-squad-PmQAm-codeD002 subs-squad-PQA-codeD002 
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 8 > logs/KnowledgeConflict/${dataset}_code002_8shot.log
# done


# for dataset in subs-squad-QA-codeD002 subs-squad-QAm-codeD002
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/KnowledgeConflict/${dataset}_code002_16shot.log
# done


# for dataset in subs-nq-PmQAm-textD001
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine text-davinci-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --continue_from 7968 \
#     --shots 4 > logs/KnowledgeConflict/${dataset}_textD001_4shot_part2.log
# done


# for dataset in subs-nq-PmQAm-textC001
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine text-curie-001 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 4 > logs/KnowledgeConflict/${dataset}_textC001_4shot.log
# done


# for dataset in IIDPromptMRQABioASQDev IIDPromptMRQADROPDev IIDPromptMRQADuoRCDev IIDPromptMRQARACEDev IIDPromptMRQAREDev IIDPromptMRQATextbookQADev
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 8 > logs/MRQAIIDPrompt/${dataset}_code002_8shot.log
# done



# for dataset in mrqa-nq-train mrqa-squad-train mrqa-newsqa-train
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 16 > logs/MRQA-train-orig/${dataset}_code002_16shot.log
# done


# for dataset in mrqa-nq-train mrqa-squad-train
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 16 \
#     --shots 16 > logs/MRQA-train-orig/${dataset}_code002_16shot.log
# done

# for dataset in MinPrompt2MRQABioASQDev MinPrompt2MRQADROPDev MinPrompt2MRQADuoRCDev MinPrompt2MRQAHotpotQADev MinPrompt2MRQANaturalQuestionsDev MinPrompt2MRQANewsQADev MinPrompt2MRQARACEDev MinPrompt2MRQAREDev MinPrompt2MRQASearchQADev MinPrompt2MRQASQuADDev MinPrompt2MRQATextbookQADev MinPrompt2MRQATriviaQADev
# do 
#     python -u cot.py \
#     --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
#     --engine code-davinci-002 \
#     --task $dataset \
#     --prompt_source $dataset \
#     --prompt_method fewshot \
#     --print \
#     --maxlen 32 \
#     --shots 48 > logs/MRQAMinPrompt2/${dataset}_code002_48shot.log
# done


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task subqa-overall-passage \
# --prompt_source subqa-overall-passage \
# --prompt_method fewshot-cot \
# --print \
# --maxlen 128 \
# --shots 16 > logs/subqa-overall-passage_code002_16shot_cot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task ambigqa \
# --prompt_source ambigqa \
# --prompt_method fewshot-cot \
# --print \
# --maxlen 128 \
# --shots 16 > logs/ambigqa_code002_16shot_cot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task triviaqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hotpotqa_to_triviaqa_code002_16shot.log


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task nq \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hotpotqa_to_nq_code002_16shot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task webqsp \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hotpotqa_to_webqsp_code002_16shot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task strategyqa \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hotpotqa_to_strategyqa_code002_16shot.log


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task boolq \
# --prompt_source hotpotqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hotpotqa_to_boolq_code002_16shot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task strategyqa \
# --prompt_source strategyqa \
# --prompt_method fewshot-cot \
# --print \
# --maxlen 128 \
# --shots 16 > logs/strategyqa_code002_16shot_cot.log



# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task freebaseqa \
# --prompt_source freebaseqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/freebaseqa_code002_16shot.log


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task hybridqa \
# --prompt_source hybridqa \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/hybridqa_code002_16shot.log


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task hotpotqa-shortcot \
# --prompt_source hotpotqa-shortcot \
# --prompt_method fewshot-cot \
# --print \
# --maxlen 128 \
# --shots 16 > logs/hotpotqa_short_cot_code002_16shot_cot.log




# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task squad \
# --prompt_source squad \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/squad_code002_16shot.log


# python -u cot.py \
# --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo \
# --engine code-davinci-002 \
# --task boolq \
# --prompt_source boolq \
# --prompt_method fewshot \
# --print \
# --maxlen 32 \
# --shots 16 > logs/boolq_code002_16shot.log



## zeroshot-step
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_text001_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_text002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_code002_zeroshot_step.log


## zeroshot-step 
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task gsm8k --print --prompt zeroshot-step --maxlen 256 --extract > logs/gsm8k_text001_zeroshot_step.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task gsm8k --print --prompt zeroshot-step --maxlen 256 --extract > logs/gsm8k_text002_zeroshot_step.log
# python -u cot2.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task gsm8k --print --prompt zeroshot-boost --maxlen 256 > logs/gsm8k_code002_zeroshot_boost.log


## zeroshot-step 
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_text001_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_text002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_code002_zeroshot_step.log

### SingleEq
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-001 --task singleeq --print --prompt zeroshot-step --maxlen 256 --extract > logs/singleeq_text001_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task singleeq --print --prompt zeroshot-step --maxlen 256 --extract > logs/singleeq_code002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task multiarith --print --prompt zeroshot-step --maxlen 256 --extract > logs/multiarith_code002_zeroshot_step.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine text-davinci-002 --task singleeq --print --prompt zeroshot-step --maxlen 256 --extract > logs/singleeq_text002_zeroshot_step.log


## AQUA-advanced
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua-advanced --print --prompt zeroshot --maxlen 64 --extract > logs/aqua_advanced_code002_zeroshot.log
# python -u cot.py --apikey sk-mUmScf7buhNb1ctzo8aEMAYZXaCKsdyed4WAHWlo --engine code-davinci-002 --task aqua-advanced --print --prompt zeroshot-step --maxlen 256 --extract > logs/aqua_advanced_code002_zeroshot_step.log

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
# from ambigqa_evaluate_script import QAPairEvaluation
from transformers import GPT2TokenizerFast
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

import random
random.seed(2022)

subset_mappings = {
    "nq": "/home/sichenglei/PromptQA/testsets/AllQA/nq.json",
    "triviaqa": "/home/sichenglei/PromptQA/testsets/AllQA/triviaqa.json",
    "squad": "/home/sichenglei/PromptQA/testsets/squad.json",
    "hotpotqa": "/home/sichenglei/PromptQA/testsets/AllQA/hotpotqa.json",
    "webq": "/home/sichenglei/PromptQA/testsets/webq.json",
    "cwq": "/home/sichenglei/PromptQA/testsets/cwq.json",
    "qampari": "/home/sichenglei/PromptQA/testsets/qampari.json",
    "boolq": "/home/sichenglei/PromptQA/testsets/boolq.json",
    "hotpotqa-shortcot": "/home/sichenglei/PromptQA/testsets/hotpot_cot.json",
    "webqsp": "/home/sichenglei/PromptQA/testsets/webqsp.json",
    "strategyqa": "/home/sichenglei/PromptQA/testsets/strategyqa.json",
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
    "nq-train": "/home/sichenglei/PromptQA/testsets/nq_dev.json",
    "hotpotqa-train": "/home/sichenglei/PromptQA/testsets/hotpotqa_train.json",
    "webqsp-train": "/home/sichenglei/PromptQA/testsets/webqsp_train.json",
    "reclor": "/home/sichenglei/PromptQA/testsets/reclor.json",
    "race": "/home/sichenglei/PromptQA/testsets/AllQA/race.json",
    "boolq-rc": "/home/sichenglei/PromptQA/testsets/boolq-rc.json",
    "nq-dpr": "/home/sichenglei/PromptQA/testsets/nq-dpr.json",
    "subqa-overall": "/home/sichenglei/PromptQA/testsets/subqa-overall.json",
    "subqa-sub1": "/home/sichenglei/PromptQA/testsets/subqa-sub1.json",
    "subqa-sub2": "/home/sichenglei/PromptQA/testsets/subqa-sub2.json",
    "subqa-goldsub1": "/home/sichenglei/PromptQA/testsets/subqa-goldsub1.json",
    "subqa-goldqa1": "/home/sichenglei/PromptQA/testsets/subqa-goldqa1.json",
    "subqa-step1": "/home/sichenglei/PromptQA/testsets/subqa-step1.json",
    "subqa-step2": "/home/sichenglei/PromptQA/testsets/subqa-step2.json",
    "subqa-overall-passage": "/home/sichenglei/PromptQA/testsets/subqa-overall-passage.json",
    "subqa-all10": "/home/sichenglei/PromptQA/testsets/subqa-all10.json",
    # MinPrompt-MRQA
    "MinPromptMRQABioASQDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQABioASQDev.json",
    "MinPromptMRQADROPDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQADROPDev.json",
    "MinPromptMRQADuoRCDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQADuoRCDev.json",
    "MinPromptMRQAHotpotQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQAHotpotQADev.json",
    "MinPromptMRQANaturalQuestionsDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQANaturalQuestionsDev.json",
    "MinPromptMRQANewsQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQANewsQADev.json",
    "MinPromptMRQARACEDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQARACEDev.json",
    "MinPromptMRQAREDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQAREDev.json",
    "MinPromptMRQASearchQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQASearchQADev.json",
    "MinPromptMRQASQuADDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQASQuADDev.json",
    "MinPromptMRQATextbookQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQATextbookQADev.json",
    "MinPromptMRQATriviaQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPromptMRQATriviaQADev.json",
    # MinPrompt2
    "MinPrompt2MRQABioASQDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQABioASQDev.json",
    "MinPrompt2MRQADROPDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQADROPDev.json",
    "MinPrompt2MRQADuoRCDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQADuoRCDev.json",
    "MinPrompt2MRQAHotpotQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQAHotpotQADev.json",
    "MinPrompt2MRQANaturalQuestionsDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQANaturalQuestionsDev.json",
    "MinPrompt2MRQANewsQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQANewsQADev.json",
    "MinPrompt2MRQARACEDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQARACEDev.json",
    "MinPrompt2MRQAREDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQAREDev.json",
    "MinPrompt2MRQASearchQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQASearchQADev.json",
    "MinPrompt2MRQASQuADDev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQASQuADDev.json",
    "MinPrompt2MRQATextbookQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQATextbookQADev.json",
    "MinPrompt2MRQATriviaQADev": "/home/sichenglei/PromptQA/testsets/mrqa/MinPrompt2MRQATriviaQADev.json",
    ## MRQA IIDPrompt for OOD datasets
    "IIDPromptMRQABioASQDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQABioASQDev.json",
    "IIDPromptMRQADROPDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQADROPDev.json",
    "IIDPromptMRQADuoRCDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQADuoRCDev.json",
    "IIDPromptMRQARACEDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQARACEDev.json",
    "IIDPromptMRQAREDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQAREDev.json",
    "IIDPromptMRQATextbookQADev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQATextbookQADev.json",
    "IIDPromptMRQASQuADDev": "/home/sichenglei/PromptQA/testsets/mrqa/IIDPromptMRQASQuADDev.json",
    ## MRQA sampled train for memorization analysis,
    "mrqa-nq-train": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/mrqa-nq-origqa.json",
    "mrqa-squad-train": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/mrqa-squad-origqa.json",
    "subs-nq-PmQAm-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-PmQAm-codeD002.json",
    "subs-nq-PQA-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-PQA-codeD002.json",
    "subs-nq-QA-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-QA-codeD002.json",
    "subs-nq-QAm-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-QAm-codeD002.json",
    "subs-squad-PmQAm-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-squad-PmQAm-codeD002.json",
    "subs-squad-PQA-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-squad-PQA-codeD002.json",
    "subs-squad-QA-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-squad-QA-codeD002.json",
    "subs-squad-QAm-codeD002": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-squad-QAm-codeD002.json",
    "subs-nq-PmQAm-textD001": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-PmQAm-textD001.json",
    "subs-nq-PmQAm-textC001": "/home/sichenglei/PromptQA/testsets/KnowledgeConflict/subs-nq-PmQAm-textC001.json",
    ## Entity-Substituted,
    "mrqa-nq-train-subs": "/home/sichenglei/PromptQA/testsets/mrqa-nq-train-subs.json",
    ## GLUE
    "mnli": "/home/sichenglei/PromptQA/testsets/GLUE/mnli_matched.json",
    "mnli-hans-format": "/home/sichenglei/PromptQA/testsets/GLUE/mnli_matched.json",
    "mnli-mm":  "/home/sichenglei/PromptQA/testsets/GLUE/mnli_mismatched.json",
    "qnli": "/home/sichenglei/PromptQA/testsets/GLUE/qnli.json",
    "qqp": "/home/sichenglei/PromptQA/testsets/GLUE/qqp.json",
    "rte": "/home/sichenglei/PromptQA/testsets/GLUE/rte.json",
    "sst2": "/home/sichenglei/PromptQA/testsets/GLUE/sst2.json",
    "mrpc": "/home/sichenglei/PromptQA/testsets/GLUE/qqp_to_mrpc.json",
    "snli":  "/home/sichenglei/PromptQA/testsets/GLUE/snli.json",
    ## AdvGLUE
    "adv-mnli": "/home/sichenglei/PromptQA/testsets/AdvGLUE/mnli_matched.json",
    "adv-mnli-mm":  "/home/sichenglei/PromptQA/testsets/AdvGLUE/mnli_mismatched.json",
    "adv-qnli": "/home/sichenglei/PromptQA/testsets/AdvGLUE/qnli.json",
    "adv-qqp": "/home/sichenglei/PromptQA/testsets/AdvGLUE/qqp.json",
    "adv-rte": "/home/sichenglei/PromptQA/testsets/AdvGLUE/rte.json",
    "adv-sst2": "/home/sichenglei/PromptQA/testsets/AdvGLUE/sst2.json",
    ## Contrast Sets
    "imdb_orig": "/home/sichenglei/PromptQA/testsets/ContrastSet/imdb_orig.json",
    "imdb_contrast": "/home/sichenglei/PromptQA/testsets/ContrastSet/imdb_contrast.json",
    "boolq_orig": "/home/sichenglei/PromptQA/testsets/ContrastSet/boolq_orig.json",
    "boolq_contrast": "/home/sichenglei/PromptQA/testsets/ContrastSet/boolq_contrast.json",
    "quoref_orig": "/home/sichenglei/PromptQA/testsets/ContrastSet/quoref_orig.json",
    "quoref_contrast": "/home/sichenglei/PromptQA/testsets/ContrastSet/quoref_contrast.json",
    "mctaco_orig": "/home/sichenglei/PromptQA/testsets/ContrastSet/mctaco_orig.json",
    "mctaco_contrast": "/home/sichenglei/PromptQA/testsets/ContrastSet/mctaco_contrast.json",
    ## SQuAD OOD
    "squad_nyt": "/home/sichenglei/PromptQA/testsets/SQuAD_OOD/nyt.json",
    "squad_reddit": "/home/sichenglei/PromptQA/testsets/SQuAD_OOD/reddit.json",
    "squad_amazon": "/home/sichenglei/PromptQA/testsets/SQuAD_OOD/amazon.json",
    ## NLI OOD
    "nli_mnli": "/home/sichenglei/PromptQA/testsets/NLI_OOD/mnli.json",
    "nli_mnli-mm":  "/home/sichenglei/PromptQA/testsets/NLI_OOD/mnli_mm.json",
    "nli_qnli": "/home/sichenglei/PromptQA/testsets/NLI_OOD/qnli.json",
    "nli_qqp": "/home/sichenglei/PromptQA/testsets/NLI_OOD/qqp.json",
    "nli_rte": "/home/sichenglei/PromptQA/testsets/NLI_OOD/rte.json",
    "nli_sst2": "/home/sichenglei/PromptQA/testsets/NLI_OOD/sst2.json",
    "nli_mrpc": "/home/sichenglei/PromptQA/testsets/NLI_OOD/mrpc.json",
    "nli_snli":  "/home/sichenglei/PromptQA/testsets/NLI_OOD/snli.json",
    "nli_scitail":  "/home/sichenglei/PromptQA/testsets/NLI_OOD/scitail.json",
    "nli_wnli":  "/home/sichenglei/PromptQA/testsets/NLI_OOD/wnli.json",
    ## HANS
    "hans_lex_ent": "/home/sichenglei/PromptQA/testsets/Spurious/hans_lex_ent.json",
    "hans_lex_non": "/home/sichenglei/PromptQA/testsets/Spurious/hans_lex_non.json",
    "hans_sub_ent": "/home/sichenglei/PromptQA/testsets/Spurious/hans_sub_ent.json",
    "hans_sub_non": "/home/sichenglei/PromptQA/testsets/Spurious/hans_sub_non.json",
    "hans_con_ent": "/home/sichenglei/PromptQA/testsets/Spurious/hans_con_ent.json",
    "hans_con_non": "/home/sichenglei/PromptQA/testsets/Spurious/hans_con_non.json",
    "paws": "/home/sichenglei/PromptQA/testsets/Spurious/paws.json",
    "mnli_half_bias": "/home/sichenglei/PromptQA/testsets/Spurious/MNLI_half_bias.json",
    "mnli_full_bias": "/home/sichenglei/PromptQA/testsets/Spurious/MNLI_full_bias.json",
    ## WinoBias
    "winobias_anti_type1":  "/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type1.json",
    "winobias_anti_type2":  "/home/sichenglei/PromptQA/testsets/WinoBias/anti_stereotyped_type2.json",
    "winobias_pro_type1": "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type1.json",
    "winobias_pro_type1_32shots": "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type1_32shots.json",
    "winobias_pro_type2": "/home/sichenglei/PromptQA/testsets/WinoBias/pro_stereotyped_type2.json",
    "winobias_balanced_prompt": "/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt.json",
    "winobias_balanced_prompt_32shots": "/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_32shots.json",
    "winobias_balanced_prompt_pro_at_end": "/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_pro_at_end.json",
    "winobias_balanced_prompt_anti_at_end": "/home/sichenglei/PromptQA/testsets/WinoBias/balanced_prompt_anti_at_end.json",
    "bbq": "/home/sichenglei/PromptQA/testsets/BBQ/bbq.json",
    "bbq_disambig": "/home/sichenglei/PromptQA/testsets/BBQ/bbq_disambig.json",
    "bbq_ambig_neutral": "/home/sichenglei/PromptQA/testsets/BBQ/bbq_ambig_neutral.json",
    "bbq_ambig_pro": "/home/sichenglei/PromptQA/testsets/BBQ/bbq_ambig_pro.json",
    "bbq_ambig_anti": "/home/sichenglei/PromptQA/testsets/BBQ/bbq_ambig_anti.json",
    ## Knowledge Updating
    "fever_orig_full": "/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_orig_test_full.json",
    "zsre_orig_full": "/home/sichenglei/PromptQA/testsets/KnowUpdate/zsre_orig_test_full.json",
    ## Retrieval
    "contriever_top5_nq": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top5_nq.json",
    "contriever_top10_nq": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top10_nq.json",
    "contriever_top5_triviaqa": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top5_triviaqa.json",
    "contriever_top10_triviaqa": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top10_triviaqa.json",
    "contriever_top5_squad": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top5_squad.json",
    "contriever_top10_squad": "/home/sichenglei/PromptQA/testsets/retrieval/contriever_top10_squad.json",
    "contriever_top5_nq_pqa": "/home/sichenglei/PromptQA/testsets_backup/retrieval/contriever_top5_nq_pqa.json",
    "contriever_top5_squad_pqa": "/home/sichenglei/PromptQA/testsets_backup/retrieval/contriever_top5_squad_pqa.json",
    "contriever_topt_triviaqa_pqa": "/home/sichenglei/PromptQA/testsets_backup/retrieval/contriever_top5_triviaqa_pqa.json",
    ## Knowledge Update 
    "fever_orig_irrelevant": "/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_orig_irrelevant.json",
    "fever_edited_test": "/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_edited_test.json",
    "fever_edited_irrelevant": "/home/sichenglei/PromptQA/testsets/KnowUpdate/fever_edited_irrelevant.json",
    "zsre_orig_irrelevant": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_orig_irrelevant.json",
    "zsre_edited_test": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_test.json",
    "zsre_edited_irrelevant": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdate/zsre_edited_irrelevant.json",
    ## Knowledge Update Balanced
    "fever_orig_irrelevant_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_orig_irrelevant.json",
    "fever_edited_test_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_test.json",
    "fever_edited_irrelevant_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/fever_edited_irrelevant.json",
    "zsre_orig_irrelevant_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_orig_irrelevant.json",
    "zsre_edited_test_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_edited_test.json",
    "zsre_edited_irrelevant_balance": "/home/sichenglei/PromptQA/testsets_backup/KnowUpdateBalance/zsre_edited_irrelevant.json",
    "fever_orig_irrelevant_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_orig_irrelevant_all_balance.json",
    "fever_edited_test_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_edited_test_all_balance.json",
    "fever_edited_irrelevant_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/fever_edited_irrelevant_all_balance.json",
    "zsre_orig_irrelevant_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_orig_irrelevant_all_balance.json",
    "zsre_edited_test_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_edited_test_all_balance.json",
    "zsre_edited_irrelevant_all_balance": "/home/sichenglei/PromptQA/testsets/KnowUpdateBalance/zsre_edited_irrelevant_all_balance.json",
    ## MMLU
    "mmlu_astronomy": "/home/sichenglei/PromptQA/testsets/MMLU/astronomy.json",
    "mmlu_nutrition": "/home/sichenglei/PromptQA/testsets/MMLU/nutrition.json",
    "mmlu_policy": "/home/sichenglei/PromptQA/testsets/MMLU/us_foreign_policy.json",
    "mmlu_astronomy_top10": "/home/sichenglei/PromptQA/testsets/MMLU/astronomy_contriever_top10.json",
    "mmlu_nutrition_top10": "/home/sichenglei/PromptQA/testsets/MMLU/nutrition_contriever_top10.json",
    "mmlu_policy_top10": "/home/sichenglei/PromptQA/testsets/MMLU/us_foreign_policy_contriever_top10.json",
    "mmlu_astronomy_top10_qonly": "/home/sichenglei/PromptQA/testsets/MMLU/astronomy_contriever_top10_qonly.json",
    "mmlu_nutrition_top10_qonly": "/home/sichenglei/PromptQA/testsets/MMLU/nutrition_contriever_top10_qonly.json",
    "mmlu_policy_top10_qonly": "/home/sichenglei/PromptQA/testsets/MMLU/us_foreign_policy_contriever_top10_qonly.json",
}

train_for_inference_mappings = {
    "nq": "/home/sichenglei/PromptQA/testsets/nq_dev.json",
    "hotpotqa": "/home/sichenglei/PromptQA/testsets/hotpotqa_train.json",
    "webqsp": "/home/sichenglei/PromptQA/testsets/webqsp_train.json",
}

# single_ans_qa = ["nq", "triviaqa", "squad", "hotpotqa", "webq", "boolq", "hotpotqa-shortcot", "cwq"]

answer_extract_mapping = {
    "nq": answer_extract_textqa,
    "triviaqa": answer_extract_textqa,
    "squad": answer_extract_textqa,
    "hotpotqa": answer_extract_textqa,
}


def PromptStep(args, prompt, temp):
    if "code" in args.engine:
        stoplist = ["<|endoftext|>", "\n\n"]
    else:
        stoplist = ["<|endoftext|>", "\n\n\n"]
    
    # if len(prompt.split()) > 7500 and "code" in args.engine:
    #     prompt = ' '.join(prompt.split()[-7500 : ])
    
    # # print (len(prompt.split()))
    # if len(prompt.split()) > 600 and "text" in args.engine:
    #     prompt = ' '.join(prompt.split()[-600 : ])

    tokenized = gpt_tokenizer.convert_tokens_to_ids(gpt_tokenizer.tokenize(prompt))
    if "code" in args.engine and len(tokenized) > 7900:
        return "", prompt
    
    if "text" in args.engine and len(tokenized) >= 2000:
        return "", prompt
    
    if args.engine == "davinci" and len(tokenized) >= 2040:
        print ("len tokenized: ", len(tokenized))
        prompt = gpt_tokenizer.decode(tokenized[-2040 : ])


    if "mmlu" in args.task and args.save_prob:
        logprobs = 4
    else:
        logprobs = 1
    ## A Single Prompt Step
    response = None
    while response is None:
        try:
            response = openai.Completion.create(
                engine=args.engine,
                prompt=prompt,
                max_tokens=args.maxlen,
                logprobs=logprobs,
                temperature=temp,
                stream=False,
                stop=stoplist
            )
        except:
            sleep(10)
            continue
   
    # print (response)
    if args.engine == "code-davinci-002" or "mmlu" in args.task:
        top_probs = []
        top_log_probs = response['choices'][0]["logprobs"]["top_logprobs"][0]
        for t in range(len(response['choices'][0]["logprobs"]["tokens"])):
            if response['choices'][0]["logprobs"]["tokens"][t] == "\n":
                break
            top_probs.append(response['choices'][0]["logprobs"]["token_logprobs"][t])
    else:
        top_probs = []
        top_tokens = []
        for t in range(len(response['choices'][0]["logprobs"]["tokens"])):
            if response['choices'][0]["logprobs"]["tokens"][t] == "\n":
                continue
            elif response['choices'][0]["logprobs"]["tokens"][t] == "<|endoftext|>":
                break
            top_probs.append(response['choices'][0]["logprobs"]["token_logprobs"][t])
            top_tokens.append(response['choices'][0]["logprobs"]["tokens"][t])
        # print (top_tokens)
    perplexity = np.exp((np.mean(top_probs)))
    output = response['choices'][0]["text"].strip()

    if args.extract:  
        prompt += output + "\n"
        prompt += "Therefore, the final answer is "

        response = openai.Completion.create(
            engine=args.engine,
            prompt=prompt,
            max_tokens=32,
            logprobs=1,
            temperature=temp,
            stream=False,
            stop=stoplist
        )
        
        output = response['choices'][0]["text"].strip()
    
    if "mmlu" in args.task and args.save_prob:
        return output, prompt, (top_log_probs, perplexity)
    else:
        return output, prompt, (top_probs, perplexity)

def SelfConPrompt(args, counter, prompt, eg):
    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        print (prompt)
    
    all_outputs = []
    ## self-consistency prompting
    ## we sample 10 different outputs with temperature 0.7
    for i in range(10):
        output, newprompt, _ = PromptStep(args, prompt, temp=0.7)
        ans = answer_extract_mapping[args.task](output)
        
        if args.print:
            print ("\nOutput #{}".format(str(i+1)))
            print (output)
            print ("\nExtracted answer string: ", ans)
        
        ## exclude no-answer cases
        if ans is not None:
            all_outputs.append(ans)
    
    final_ans = most_common(all_outputs)
    gold = eg["answer"]
    # match = 0
    # if type(gold) is list:
    #     for g in gold:
    #         match = max(match, answer_eval_mapping[args.task](final_ans, g))
    # else:
    #     match = answer_eval_mapping[args.task](final_ans, gold)
    em = single_ans_em(final_ans, gold)
    f1 = single_ans_f1(final_ans, gold)
    
    if args.print:
        print ("\n\nQuestion #{} Summary: ".format(str(counter)))
        print ("All predicted answers: ", all_outputs)
        print ("Final prediction: ", final_ans)
        print ("Prob (frequency): ", all_outputs.count(final_ans) / len(all_outputs))
        print ("Gold answer: ", gold)
        print ("EM: ", em)
        print ("F1: ", f1)
        print ("\n\n")
    
    return em, f1


def SinglePrompt(args, counter, prompt, eg):
    ## greedy decoding by default
    output, newprompt, probs = PromptStep(args, prompt, temp=0.)
        
    if output.lower().strip() == "unanswerable":
        output_ = ''
    else:
        output_ = output

    gold = eg["answer"]
    em = 0
    f1 = 0
    
    orig_output = output[:]
    prefix = "answer is "
    if prefix in output:
        idx = output.rfind(prefix)
        output = output[idx + len(prefix) : ]

    if args.task in ["ambigqa"]:
        output = list(set(output.split("; ")))
        prediction = {}
        prediction[eg["id"]] = output
        prdiction = [prediction]
        reference = [eg]
        evaluation = QAPairEvaluation(reference, prediction)
        # evaluation.print_all_metrics()
        # em = int(get_exact_match(output, gold))
        # f1 = get_f1(output, gold)
        em = f1 = evaluation.get_metric("F1 answer")
        output = "; ".join(output)
    else:
        em = single_ans_em(output_, gold)
        f1 = single_ans_f1(output_, gold)
        
    if args.print:
        print ("**********     Question #{}    **********".format(str(counter)))
        print (newprompt + orig_output)
        # if args.task == "ambigqa":
        #     print ("Extracted predictions: ", output)
        print ("\nGold answer: ", gold)
        if "orig_answer" in eg:
            print ("Orig answer: ", eg["orig_answer"])
        if "mrqa-" in args.task:
            print ("ID: ", eg["id"])
        if args.task in ["boolq_orig", "boolq_contrast", "bbq", "fever_orig_full", "zsre_orig_full"]:
            print ("ID: ", eg["id"])
        if "contriever" in args.task:
            print ("has_answer: ", eg["has_answer"])
        if args.save_prob:
            if "mmlu" in args.task:
                for k,v in probs[0].items():
                    probs[0][k] = np.exp(v)
                print ("token_probs: ", probs[0])
            else:
                print ("token_probs: ", [np.exp(p) for p in probs[0]])
            print ("LM Prob: ", probs[1])
        print ("EM ", em)
        print ("F1 ", f1)
        print ("\n\n")
    
    return em, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apikey', type=str, required=True, help='api key; https://openai.com/api/')
    parser.add_argument('--engine', type=str, default='davinci', help='api engine; https://openai.com/api/')
    parser.add_argument('--task', type=str, help='specify the task that you want to evaluate')
    parser.add_argument('--prompt_source', type=str, help='specify the where the prompt demos should come from')
    parser.add_argument('--prompt_method', type=str, default=None, help='specify the prompting method', choices=["zeroshot", "zeroshot-step", "fewshot", "fewshot-selfcon", "fewshot-cot", "fewshot-cot-selfcon"])
    parser.add_argument('--print', default=False, action='store_true', help='Whether to print out every prompt')
    parser.add_argument('--extract', default=False, action='store_true', help='Whether to add an additional answer extraction step')
    parser.add_argument('--subset', default=False, action='store_true', help='Whether to use a small subset for debugging')
    parser.add_argument('--subset_size', type=int, default=32, help='how many examples to sample for quick evaluation')
    parser.add_argument('--maxlen', type=int, default=256, help='max number of tokens to be generated')
    parser.add_argument('--shots', type=int, default=0, help='how many demos to use in the prompt')
    parser.add_argument('--no_unanswerable', default=False, action='store_true', help='Whether to filter out unanswerable questions in the demo')
    parser.add_argument('--label_shuffle', default=False, action='store_true', help='Whether to shuffle the gold labels')
    parser.add_argument('--save_prob', default=False, action='store_true', help='Whether to save top token logprobs and confidence')
    parser.add_argument('--continue_from', type=int, default=0, help='evaluate on part of test set, starting from this index')

    args = parser.parse_args()
    openai.api_key = args.apikey

    all_em = 0
    all_f1 = 0

    if args.task in subset_mappings:
        ## load test set
        task_dir = subset_mappings[args.task]
        with open(task_dir, "r") as f:
            data = json.load(f)
        test_set = data["testset"]
        if args.subset:
            test_set = test_set[ : args.subset_size]
        print ("Size of test set:", len(test_set))

        ## load demos
        if args.prompt_source != args.task:
            task_dir = subset_mappings[args.prompt_source]
            with open(task_dir, "r") as f:
                data = json.load(f)
        ## filter out unanswerables in demos
        if "timeqa" in args.task and args.no_unanswerable:
            data["demos"] = [eg for eg in data["demos"] if eg["answer"][0] != '']
        
        if "text" in args.engine:
            ## select shortest prompts to fit into the context
            demos_len = [len(d["question"].split()) for d in data["demos"]]
            demos_idx = np.argsort(demos_len)
            demos = []
            for idx in demos_idx[ : args.shots]:
                demos.append(data["demos"][idx])
        else:
            demos = data["demos"][ : args.shots]
        
        ## condense MNLI labels for HANS
        if "hans" in args.task:
            ent = []
            non = []
            for di in range(len(demos)):
                demos[di]["question"] = "\n".join(demos[di]["question"].split("\n")[:2] + ["Can the second sentence be inferred from the first sentence?"])
                if demos[di]["answer"] != "entailment":
                    demos[di]["answer"] = ["no"]
                    non.append(demos[di])
                else:
                    demos[di]["answer"] = ["yes"]
                    ent.append(demos[di])
            if "text" in args.engine:
                demos = ent[:8] + non[:8]
            else:
                demos = ent[:16] + non[:16]
            random.shuffle(demos)
        print ("#shots: ", len(demos))

    else:
        print ("Task is out of our data collection")
        return 

    test_set = test_set[ args.continue_from : ]
    counter = args.continue_from
    demos_questions = [d["question"].strip() for d in demos]
    for eg in tqdm(test_set):
        
        if eg["question"].strip() in demos_questions:
            continue

        counter += 1

        # instruction = "Please answer the following questions correctly. Leave it empty if you don't know the answer or if the question is unanswerable. \n"
        # instruction = "Please answer the following questions correctly. \n"
        prompt = ""
        # prompt += instruction + "\n"
        
        ## few-shot demo
        if args.prompt_method in ["fewshot", "fewshot-selfcon", "fewshot-cot", "fewshot-cot-selfcon"]:
            if args.label_shuffle:
                labels = [d["answer"] for d in demos]
                random.shuffle(labels)
                for i in range(len(labels)):
                    demos[i]["answer"] = labels[i]

            for demo in demos:
                prompt += demo["question"] + "\n"
                answer = demo["answer"]
                
                if args.prompt_source in ["ambigqa"]:
                    # answer = [x for xs in answer for x in xs]
                    answer = [qa[0] for qa in answer]
                    answer = list(set(answer))
                    answer = "; ".join(answer)

                elif type(answer) is list:
                    answer = answer[0] # only pick one answer for each demo
                
                if answer == "":
                    answer = "unanswerable"

                if args.task == "strategyqa" and args.prompt_source == "boolq":
                    if answer == "True":
                        answer = "yes"
                    elif answer == "False":
                        answer = "no"
                
                if args.task == "boolq-rc" and args.prompt_source == "boolq":
                    if answer == "True":
                        answer = "yes"
                    elif answer == "False":
                        answer = "no"

                
                if args.prompt_method in ["fewshot", "fewshot-selfcon"]:
                    ## without cot
                    if args.task not in ["subqa-step1"]:
                        prompt += "Answer: " + answer.strip() + "\n\n"
                    else:
                        prompt += answer.strip() + "\n\n"
                elif args.prompt_method == "fewshot-cot":
                    ## with cot
                    # prompt += "Answer: Let’s think step by step. " + demo["cot"] + "\n"
                    # prompt += "Answer: This question can be decomposed into a few sub-questions: " + demo["cot"] + "\n"
                    if args.task not in ["subqa-goldsub1", "subqa-goldqa1"]:
                        prompt += "Answer: "

                    # print (demo["answer"])
                    if args.prompt_source and len(demo["answer"]) > 1:
                        answer = []
                        dq = demo["annotations"][0]["qaPairs"]
                        prompt += "There are {} possible interpretations of this question:\n".format(str(len(dq)))
                        for q in range(len(dq)):
                            prompt += str(q+1) + '. ' + dq[q]["question"] + ' The answer is ' + dq[q]["answer"][0] + '\n'
                            answer.append(dq[q]["answer"][0])
                        answer = "; ".join(answer)
                        # prompt += "\n"
                        prompt += "Therefore, the final answer is " + answer.strip() + "\n\n"
                    elif "cot" in demo:
                        prompt += demo["cot"]
                        prompt += "\n"
                        prompt += "Therefore, the final answer is " + answer.strip() + "\n\n"
                    else:
                        prompt += answer.strip() + "\n\n"
        
        ## current test instance 
        if len(eg["question"].split()) > 400 and "text" in args.engine:
            eg["question"] = ' '.join(eg["question"].split()[-400 : ])
        
        if "hans" in args.task:
            eg["question"] = "\n".join(eg["question"].split("\n")[:2] + ["Can the second sentence be inferred from the first sentence?"])
            if args.shots == 0:
                eg["question"] = eg["question"].split("\n")[0] + "\n" + "Does it mean " + eg["question"].split("\n")[1][0].lower() + eg["question"].split("\n")[1][1:] + " (no or yes)"
            if "entailment" not in eg["answer"]:
                eg["answer"] = ["no"]
            else:
                eg["answer"] = ["yes"]
        
        if "mnli" in args.task and args.shots == 0:
            eg["question"] += " (choose from: entailment, neutral, or contradiction)"
        # if "winobias" in args.task:
        #     eg["question"] = "Men and women are equally capable. We should not discriminate any gender.\n" + eg["question"] 
        
        if args.prompt_source == "bbq_ambig_pro":
            eg["question"] = "Note that, we should treat people from different socioeconomic status, sexual orientations, religions, races, physical appearances, nationalities, gender identities, disabilities, and ages equally. "\
                + "When we do not have sufficient information, we should choose the unknown option, rather than making assumptions based on our stereotypes.\n" + eg["question"]

        prompt += eg["question"]  + "\n"
        
        if args.engine == "davinci" and "mmlu" in args.task:
            prompt += "Answer:"
        elif args.task not in ["subqa-goldsub1", "subqa-goldqa1", "subqa-step1"]:
            prompt += "Answer: "


        # if "cot" in args.prompt_method:
        #     prompt += "Let's think step by step. "

        # if args.prompt_method in ["zeroshot-step", "fewshot-cot", "fewshot-cot-selfcon"]:
        #     prompt += "Let’s think step by step. "
        
        # if args.prompt_method == "fewshot-cot":
        #     prompt += "Based on the fact that "
        # else:
        #     prompt += "Answer: "
        
        if "selfcon" in args.prompt_method:
            em, f1 = SelfConPrompt(args, counter, prompt, eg)
        else:
            em, f1 = SinglePrompt(args, counter, prompt, eg)

        all_em += em 
        all_f1 += f1
    
    print ("EM: {}/{}={}%".format(all_em, counter, all_em / counter * 100))
    print ("F1: {}/{}={}%".format(all_f1, counter, all_f1 / counter * 100))
    

if __name__ == '__main__':
    main()
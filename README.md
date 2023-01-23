# Prompting GPT-3 To Be Reliable (ICLR 2023)

This is the repo that contains: 1) code to reproduce all experiment results; 2) all processed datasets used in the paper; 3) the model outputs (i.e., GPT-3 generations with different prompts) for all experiments conducted in the paper. 


## Code 

### Dependencies

* openai 
* transformers (2.9.0)
* tokenizers (0.7.0)

Note that ``transformers`` and ``tokenizers`` are optional, we mainly use it to tokenizer the input text with ``GPT2Tokenizer`` in order to measure input lengths. 

### Run Evaluation

Once you download the datasets (introduced in the next section), you can directly start running GPT-3 evaluation with ``cot.py``. You may have to change the data paths in the first part of ``cot.py`` to read data from the correct paths. We already have the data processing and GPT-3 querying code which you don't need to change. 

An example script is as follows:

```bash
# run evaluation
for dataset in nq hotpotqa
do 
    python -u cot.py \
    --apikey YOURKEY \
    --engine code-davinci-002 \
    --task $dataset \
    --prompt_source $dataset \
    --prompt_method fewshot \
    --print \
    --save_prob \
    --maxlen 32 \
    --shots 16 > logs/calibration/${dataset}_probs_code002_16shot.log
done

```

* You can see the full list of supported datasets from ``subset_mappings`` in ``cot.py``. For supported datasets, you can directly specify their names/keys in ``--task`` (like ``nq`` and ``hotpotqa``). 
* Substitute ``YOURKEY`` with your actual API key. 
* The ``--prompt_source`` argument allows you to specify where to sample the demo examples, which allows you to evaluate OOD settings where the demos come from a different dataset than the test set. 
* The ``--save_prob`` argument will record the LM probability (i.e., confidence) of the model on the predicted answer. 
* The ``--maxlen`` argument specifies the max length of the GPT-3 generation. If you use ``code-davinci-002``, the max sequence length (input + generation) is 8k; if you use ``text-davinci-002``, it's 2k. If your input (prompt) is very long, you will have to cut down the ``--maxlen`` for the generation, otherwise it will throw errors. 
* The ``--shots`` argument specifies how many demos to sample as the prompt. 
* For ``--prompt_method``, apart from standard few-shot prompting, we also support Chain-of-Thought prompting and Self-Consistency Ensemble. You can refer to the code for details. 
* We provide all evaluation scripts that we used in ``run.sh`` which you can directly use or adapt.


## Data

We have converted all datasets into the same unified format so that we can use the same evaluation script for all experiments. In each dataset json file, there are randomly sampled demo examples (which are usually sampled from the original training sets) followed by the full test sets used for evaluation. 

You can download the processed datasets from [this link](https://drive.google.com/file/d/1XfPbxJpVbeNwRTubyX-6NIY6b52LBtGi/view?usp=sharing).
If you want to add experiments on additional datasets not covered here, use ``sampler.py`` for dataset processing (you may need to add new functions to support new dataset formats). 

Note that we are using these datasets for academic research purposes only, you should check their original licenses for other use cases. 
And you should always cite the original authors of these datasets if you use them in your work. 


## GPT-3 Predictions

We recorded and share all GPT-3 output files, which can be downloaded [from here](https://drive.google.com/file/d/1Mlj8kciJzX96Sfl7iGH1M2e3V-krvOAG/view?usp=sharing). Each file contains: name of the test set, number of total examples, overall accuracy, and detailed output for all examples. In the output for each example, we include: the full prompt (i.e., exactly what feed to the model), GPT-3's prediction, the correctness of its prediction (matched with the gold reference), and in some cases we also include the probability/confidence of the model output which can be used for calibration purposes. (For the confidence, I include both the per-token probablity, as well as the LM prob of the whole answer string. More details can be found in the paper for how we obtain the confidence.)

To help you quickly get a sense of what the model predictions look like, we provide some sampled predictions in ``logs_sampled``. In each file we sampled the first 100 predictions from the test set, and we brieflt explain them here:

* ``contriever_top5_squad_code002_16shot.log``: We prepend the top-5 passages retrieved by Contriever from Wikipedia to each test question in SQuAD. 
* ``KnowledgeConflict-NQ-PmQAm_code002_8shot.log``: We prepend counterfactual passages to NQ questions and assess whether GPT-3 predicts the answer supported by the passages. (The accuracy is computed by whether the predicted answer matches the counterfactual answer, not the original correct answer.)
* ``qqp_to_paws_code002_32shot.log``: We use QQP examples as the prompt and evaluate on PAWS to assess the transfer ability (PAWS contains examples that counter the lexical overlap spurious feature). 
* ``SourcePromptMRQABioASQDev_code002_8shot.log``: We sample examples from MRQA's source domain as the prompt and evaluate on the BioASQ dataset from the target domain to assess the OOD transfer.
* ``triviaqa_probs_code002_16shot.log``:  We record the confidence of GPT-3 predictions on TriviaQA along with the answer predictions and accuracy. 

For full logs and many other datasets, check the full log repo linked above! 


## Questions? 

If you have any questions related to the code, data, predictions, or the paper, feel free to email Chengei `(sichenglei1125@gmail.com)` or open an issue. Please try to specify the problem with details so I can help you better and quicker! 

If your question is ``Can you try GPT-3 on xxx datasets/tasks for me?``, I unfortunately might have to say no because I have left Microsoft and I now have limited bandwidth and resources to run new GPT-3 experiments. But there are many datasets that I have experimented on and didn't include in the paper, so feel free to ask me anyway, I'd be happy to share results if I have them, otherwise I can point you to any related work I'm aware of. 


## Citation 

If you find our work useful, please consider citing it:
```bibtex
@inproceedings{si2022prompting,
   title={Prompting GPT-3 To Be Reliable},
   author={Chenglei Si and Zhe Gan and Zhengyuan Yang and Shuohang Wang and Jianfeng Wang and Jordan Boyd-Graber and Lijuan Wang},
   booktitle={International Conference on Learning Representations (ICLR)},   
   year={2023},
   url={https://arxiv.org/abs/2210.09150}
}
```



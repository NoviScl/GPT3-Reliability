# Prompting GPT-3 To Be Reliable 

This is the repo that contains: 1) code to reproduce all experiment results; 2) all processed datasets used in the paper; 3) the model outputs (i.e., GPT-3 generations with different prompts) for all experiments conducted in the paper. 


## Code 


## Data

We have converted all datasets into the same unified format so that we can use the same evaluation script for all experiments. In each dataset json file, there are randomly sampled demo examples (which are usually sampled from the original training sets) followed by the full test sets used for evaluation. 

You can download the processed datasets from [this link](https://drive.google.com/file/d/1XfPbxJpVbeNwRTubyX-6NIY6b52LBtGi/view?usp=sharing).
If you want to add experiments on additional datasets not covered here, use ``sampler.py`` for dataset processing (you may need to add new functions to support new dataset formats). 

Note that we are using these datasets for academic research purposes only, you should check their original licenses for other use cases. 
And you should always cite the original authors of these datasets if you use them in your work. 


## GPT-3 Predictions

We recorded and share all GPT-3 output files, which can be downloaded [from here](https://drive.google.com/file/d/1XfPbxJpVbeNwRTubyX-6NIY6b52LBtGi/view?usp=sharing). Each file contains: name of the test set, number of total examples, overall accuracy, and detailed output for all examples. In the output for each example, we include: the full prompt (i.e., exactly what feed to the model), GPT-3's prediction, the correctness of its prediction (matched with the gold reference), and in some cases we also include the probability/confidence of the model output which can be used for calibration purposes. (For the confidence, I include both the per-token probablity, as well as the LM prob of the whole answer string. More details can be found in the paper for how we obtain the confidence.)

## Questions? 

If you have any questions related to the code, data, predictions, or the paper, feel free to email Chengei `(sichenglei1125@gmail.com)` or open an issue. Please try to specify the problem with details so I can help you better and quicker! 

If your question is ``Can you try GPT-3 on xxx datasets/tasks for me?``, I unfortunately might have to say no because I have left Microsoft and I now have limited bandwidth and resources to run new GPT-3 experiments. But there are many datasets that I have experimented on and didn't include in the paper, so feel free to ask me anyway, I'd be happy to share results if I have them, otherwise I can point you to any related work I'm aware of. 


## Citation 

If you find our work useful, please consider citing it:
```bibtex
@article{si2022prompting,
   title={Prompting GPT-3 To Be Reliable},
   author={Chenglei Si and Zhe Gan and Zhengyuan Yang and Shuohang Wang and Jianfeng Wang and Jordan Boyd-Graber and Lijuan Wang},
   journal={arXiv},
   year={2022}
}
```


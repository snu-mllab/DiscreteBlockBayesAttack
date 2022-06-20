# Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization

Official PyTorch implementation of **"[Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization](https://arxiv.org/abs/2206.08575)"**, published at **ICML'22**

> **Abstract** *We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.*

## Machine Information
Below are the information about machine that authors used.
* OS: Ubuntu 16.04
* CUDA Driver Version: 460.27.04
* gcc: 5.4.0
* nvcc(CUDA): 11.2
* CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
* GPU: NVIDIA TITAN Xp

## Contents
For the initial step, use the following command to install our git repository:

```git clone --recursive https://github.com/snu-mllab/DiscreteBlockBayesAttack.git```

Our implemented code for NLP domain can be found in [nlp\_attack folder](nlp_attack)

Our implemented code for protein domain can be found in [protein\_attack folder](protein_attack)

## Citation
```
@inproceedings{leeICML22,
title = {Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization},
author = {Lee, Deokjae and Moon, Seungyong and Lee, Junhyeok and Song, Hyun Oh},
booktitle = {International Conference on Machine Learning (ICML)},
year = {2022}
}
```
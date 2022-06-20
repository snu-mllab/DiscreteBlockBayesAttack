# Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization

Official PyTorch implementation of **"[Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization](https://arxiv.org/abs/2206.08575)"**, published at **ICML'22**

Below are environments that authors used.
* OS: Ubuntu 16.04
* CUDA Driver Version: 460.27.04
* gcc: 5.4.0
* nvcc(CUDA): 11.2
* CPU: Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz
* GPU: NVIDIA TITAN Xp

This repository requires GPU to run

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
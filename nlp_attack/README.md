# Attack on NLP domain.

## Setup
1. If `gcc` do not exist in your system, please install build essential: `sudo apt-get install build-essential`
2. Create conda environment: `conda create -n nlp_atk python=3.9.7` and `conda activate nlp_atk`
3. Install textattack: `cd TextAttack;pip install -e '.[tensorflow] --extra-index-url https://download.pytorch.org/whl/cu113'`
4. Download omw-1.4: cd ..;python download.py`
## Run

### Arguments
* --recipe : full attack recipe(ex. `bayesattack-wordnet`).
* --random-seed : random seed.
* --num-examples : the number of examples to process.
* --sidx : start index of dataset.
* --pkl-dir : directory to save budget information.
* --use-sod : use this option for sod dataset sampling.
* --post-opt : use 'v3' to use the post optimization algorithm in our paper.
* --dpp-type : `dpp_posterior` for batch update via DPP.
* --max-budget-key-type : the name of target baseline to compare. this option set max query budget of our method same to target baseline. one of ['pwws','textfooler','pso','lsh','bae'].
* --max-loop : `5` for default setting. (the number of loops of BBA)
* --fit-iter : `3` for default setting. (the number of update steps in GP parameter fitting)
* --max-patience : query budget for post optimization.


### Baselines (PWWS, TextFooler, PSO)
To reproduce results of the baseline method in table 1, we produce some example commands.

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --product-space --recipe pwws --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS
```

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --product-space --recipe textfooler --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS
```

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --product-space --recipe pso --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS
```

Categories
* recipe : pwws, textfooler, pso 
* model : bert-base-uncased-mr, bert-base-uncased-ag-news, lstm-mr, lstm-ag-news, xlnet-base-cased-mr, xlnet-base-cased-ag-news

### Our Method
We provide commands to reproduce our results in table 1. The `max-patience` value used in our experiment can be found in table 7.

#### WordNet 
|Dataset|Model|Method | ASR (\%)| MR (\%)| Qrs |
|---|---|---|---|---|---|
|AG|BERT-base| PWWS| 57.1| 18.3|   367|
||    | BBA| __77.4__| __17.8__|   __217__|
||LSTM| PWWS| 78.3| 16.4|   336|
||    | BBA| __83.2__| __15.4__|   __190__|
|MR|XLNet-base| PWWS| 83.9| __14.4__|   143|
||    | BBA| __87.8__| __14.4__|    __77__|
||BERT-base| PWWS| 82.0| 15.0|   143|
||    | BBA| __88.3__| __14.6__|    __94__|
||LSTM| PWWS| __94.2__| 13.3|   132|
||    | BBA| __94.2__| __13.0__|    __67__|

BERT (AG's News, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50
```
LSTM (AG's News, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50
```
XLNet (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 100
```
BERT (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50
```
LSTM (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50
```



#### Embedding

|Dataset|Model|Method | ASR (\%)| MR (\%)| Qrs | 
|---|---|---|---|---|---|
|AG|BERT-base|   TF| 84.7| 24.9|   346|
||    | BBA| __96.0__| __18.9__|   __154__|
||LSTM|   TF| 94.9| 17.3|   228|
||    | BBA| __98.5__| __16.6__|   __142__|
|MR|XLNet-base|   TF| 95.0| 18.0|   101|
||    | BBA| __96.3__| __16.2__|    __68__|
||BERT-base|   TF| 89.2| 20.0|   115|
||    | BBA| __95.7__| __16.9__|    __67__|
||LSTM|   TF| __98.2__| 13.6|    72|
||    | BBA| __98.2__| __13.1__|    __54__|

BERT-base (AG's News, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20
```
LSTM (AG's News, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20
```
XLNet-base (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20
```
BERT-base (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20
```
LSTM (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20
```

#### HowNet

|Dataset|Model|Method | ASR (\%)| MR (\%)| Qrs |
|---|---|---|---|---|---|
|AG|BERT-base|  PSO| 67.2| 21.2| 65860|
||    |BBA| __70.8__| __15.5__|  __5176__|
||LSTM|  PSO| 71.0| 19.7| 44956|
||    | BBA| __71.9__| __13.7__|  __3278__|
|MR|XLNet-base|  PSO| __91.3__| 18.6|  4504|
||    | BBA| __91.3__| __11.7__|   __321__|
||BERT-base|  PSO| __90.9__| 17.3|  6299|
||    | BBA| __90.9__| __12.4__|   __403__|
||LSTM|  PSO| __94.4__| 15.3|  2030|
||    | BBA| __94.4__| __11.2__|   __138__|

BERT-base (AG's News, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100
```
LSTM (AG's News, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100
```
XLNet-base (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100
```
BERT-base (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100
```
LSTM (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model lstm-mr  --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100
```



### Commands for Other Datasets
We refer to commands in [exp_imdb.txt](exp_imdb.txt), [exp_yelp.txt](exp_yelp.txt), and [exp_nli.txt](exp_nli.txt).

## Citation
```
@inproceedings{leeICML22,
title = {Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization},
author = {Lee, Deokjae and Moon, Seungyong and Lee, Junhyeok and Song, Hyun Oh},
booktitle = {International Conference on Machine Learning (ICML)},
year = {2022}
}
```

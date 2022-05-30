# Attack on NLP domain.

## Setup
1. If `gcc` do not exist in your system, please install build essential: `sudo apt-get install build-essential`
2. Create conda environment: `conda create -n nlp_atk python=3.9.7` and `conda activate nlp_atk`
3. Install textattack: `cd TextAttack;pip install -e '.[tensorflow]'`
4. Install botorch: `pip install botorch`
5. Install dppy: `pip install dppy`
6. `cd ..` (go to `nlp_attack` directory)
7. `python download.py`
## Run

### Arguments
* --recipe : full attack recipe(ex. `bayesattack-wordnet`).
* --random-seed : random seed.
* --num-examples : the number of examples to process.
* --sidx : start index of dataset.
* --pkl-dir : directory to save budget information.
* --kernel-name : `categorical` for default setting.
* --search-type : include `post-opt` to do post optimization & include `sod` for sod dataset sampling.
* --block-policy : `straight` for default setting.
* --dpp-type : `dpp_posterior` for batch update via DPP.
* --max-budget-key-type : the name of target baseline to compare. this option set max query budget of our method same to target baseline.
* --max-loop : `5` for default setting.
* --max-patience : query budget for post optimization.

### Baselines (PWWS, TextFooler, PSO)
To reproduce results of the baseline method in table 1, we produce some example commands.

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe pwws --model bert-base-uncased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test
```

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe textfooler --model lstm-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test
```

```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe pso --model xlnet-base-cased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test
```

Categories
* recipe : pwws, textfooler, pso 
* model : bert-base-uncased-mr, bert-base-uncased-ag-news, lstm-mr, lstm-ag-news, xlnet-base-cased-mr, xlnet-base-cased-ag-news

### Our Method
We provide commands to reproduce our results in table 1.

#### WordNet 
BERT (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pwws --max-loop 5 --max-patience 100
```
BERT (AG's News, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pwws --max-loop 5 --max-patience 100
```
LSTM (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-wordnet --model lstm-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pwws --max-loop 5 --max-patience 100
```
LSTM (AG's News, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-wordnet --model lstm-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pwws --max-loop 5 --max-patience 100
```
XLNet (Movie Review, WordNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-wordnet --model xlnet-base-cased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pwws --max-loop 5 --max-patience 100
```

#### Embedding
BERT (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type textfooler --max-loop 5 --max-patience 50
```
BERT (AG's News, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type textfooler --max-loop 5 --max-patience 50
```
LSTM (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-embedding --model lstm-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type textfooler --max-loop 5 --max-patience 50
```
LSTM (AG's News, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-embedding --model lstm-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type textfooler --max-loop 5 --max-patience 50
```
XLNet (Movie Review, Embedding)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-embedding --model xlnet-base-cased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type textfooler --max-loop 5 --max-patience 50
```

#### HowNet
BERT (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pso --max-loop 5 --max-patience 100
```
BERT (AG's News, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pso --max-loop 5 --max-patience 100
```
LSTM (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-hownet --model lstm-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pso --max-loop 5 --max-patience 100
```
LSTM (AG's News, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-hownet --model lstm-ag-news --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pso --max-loop 5 --max-patience 100
```
XLNet (Movie Review, HowNet)
```bash
textattack attack --silent --shuffle --shuffle-seed 0 --recipe bayesattack-hownet --model xlnet-base-cased-mr --random-seed 0 --num-examples 500 --sidx 0 --pkl-dir test --kernel-name categorical --search-type post-opt_sod --block-policy straight --dpp-type dpp_posterior --max-budget-key-type pso --max-loop 5 --max-patience 100
```


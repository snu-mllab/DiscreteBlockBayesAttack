# Attack on NLP domain.

## Setup
1. If `gcc` do not exist in your system, please install build essential: `sudo apt-get install build-essential`
2. Create conda environment: `conda create -n nlp_atk python=3.9.7` and `conda activate nlp_atk`
3. conda install jsonnet -c conda-forge
4. conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
5. Install textattack: `cd TextAttack;pip install -e '.[tensorflow]'`
6. Install botorch: `pip install botorch`
7. Install dppy: `pip install dppy`
8. `cd ..` (go to `nlp_attack` directory)
9. `python download.py`
## Run

### Arguments
* --recipe : full attack recipe(ex. `bayesattack-wordnet`).
* --random-seed : random seed.
* --num-examples : the number of examples to process.
* --sidx : start index of dataset.
* --pkl-dir : directory to save budget information.
* --search-type : include `post-opt` to do post optimization & include `sod` for sod dataset sampling.
* --dpp-type : `dpp_posterior` for batch update via DPP.
* --max-budget-key-type : the name of target baseline to compare. this option set max query budget of our method same to target baseline.
* --max-loop : `5` for default setting. (the number of loops of BBA)
* --fit-iter : `3` for default setting. (the number of update steps in GP parameter fitting)
* --max-patience : query budget for post optimization.



\bottomrule
\end{tabular}
\end{adjustbox}
\end{subtable}
    \begin{subtable}[h!]{0.655 \columnwidth}
\caption{Embedding}
\centering
\begin{adjustbox}{max width=\columnwidth}
\begin{tabular}{cccccc}
    \toprule
Dataset&Model&Method & ASR (\%)& MR (\%)& Qrs \\

\midrule
AG&BERT-base&   TF& 84.7& 24.9&   346\\
&    & BBA& \textbf{96.0}& \textbf{18.9}&   \textbf{154}\\ %% 20
\cmidrule{2-6}
&LSTM&   TF& 94.9& 17.3&   228\\
&    & BBA& \textbf{98.5}& \textbf{16.6}&   \textbf{142}\\ %% 20
\midrule
MR&XLNet-base&   TF& 95.0& 18.0&   101\\
&    & BBA& \textbf{96.3}& \textbf{16.2}&    \textbf{68}\\ %% 20
\cmidrule{2-6}
&BERT-base&   TF& 89.2& 20.0&   115\\
&    & BBA& \textbf{95.7}& \textbf{16.9}&    \textbf{67}\\ %% 20
\cmidrule{2-6}
&LSTM&   TF& \textbf{98.2}& 13.6&    72\\
&    & BBA& \textbf{98.2}& \textbf{13.1}&    \textbf{54}\\ %% 20
\bottomrule
\end{tabular}
\end{adjustbox}
\end{subtable}
\begin{subtable}[h!]{0.682 \columnwidth}
    \caption{HowNet}
    \centering
    \begin{adjustbox}{max width=\columnwidth}
    \begin{tabular}{cccccc}
        \toprule
Dataset&Model&Method & ASR (\%)& MR (\%)& Qrs \\
\midrule
AG&BERT-base&  PSO& 67.2& 21.2& 65860\\
&    &BBA& \textbf{70.8}& \textbf{15.5}&  \textbf{5176}\\ %% 100
\cmidrule{2-6}
&LSTM&  PSO& 71.0& 19.7& 44956\\
&    & BBA& \textbf{71.9}& \textbf{13.7}&  \textbf{3278}\\ %% 100
\midrule
MR&XLNet-base&  PSO& \textbf{91.3}& 18.6&  4504\\
&    & BBA& \textbf{91.3}& \textbf{11.7}&   \textbf{321}\\ %% 100
\cmidrule{2-6}
&BERT-base&  PSO& \textbf{90.9}& 17.3&  6299\\
&    & BBA& \textbf{90.9}& \textbf{12.4}&   \textbf{403}\\ %% 100
\cmidrule{2-6}
&LSTM&  PSO& \textbf{94.4}& 15.3&  2030\\
&    & BBA& \textbf{94.4}& \textbf{11.2}&   \textbf{138}\\ %% 100
\bottomrule
    \end{tabular}
    \end{adjustbox}
    \end{subtable}
\end{table*}

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
|---|---|---|---|---|
Dataset|Model|Method | ASR (\%)| MR (\%)| Qrs 
|---|---|---|---|---|
AG|BERT-base| PWWS| 57.1| 18.3|   367|
|    | BBA| \textbf{77.4}| \textbf{17.8}|   \textbf{217}|
||---|---|---|---|
|LSTM| PWWS| 78.3| 16.4|   336|
|    | BBA| \textbf{83.2}| \textbf{15.4}|   \textbf{190}|
|---|---|---|---|---|
MR|XLNet-base| PWWS| 83.9| \textbf{14.4}|   143|
|    | BBA| \textbf{87.8}| \textbf{14.4}|    \textbf{77}|
||---|---|---|---|
|BERT-base| PWWS| 82.0| 15.0|   143|
|    | BBA| \textbf{88.3}| \textbf{14.6}|    \textbf{94}|
||---|---|---|---|
|LSTM| PWWS| \textbf{94.2}| 13.3|   132|
|    | BBA| \textbf{94.2}| \textbf{13.0}|    \textbf{67}|
|---|---|---|---|---|
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



### Reproduce other tables
We will add the list of commands that reproduce other tables soon.
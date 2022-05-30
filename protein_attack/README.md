# Attack on Protein Classification Dataset, EC50.

## Dependencies
for evaluation: `pytorch` `fastai` `fire` 
for bayesian optimization: `botorch` `dppy`

## Setup
1. Create conda environment: `conda create -n protein_atk python=3.9.7` and `conda activate protein_atk`
2. Install pytorch: `conda install pytorch -c pytorch`
3. Install fastai: `conda install -c fastai fastai=1.0.52`
4. Install fire: `conda install fire -c conda-forge`
5. Install botorch: `pip install botorch`
6. Install dppy: `pip install dppy`
7. Install sklearn: `pip install sklearn`
7. Download EC50 Dataset: `wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EV4ma7OclSdWCsJk_5_aiIvXTgTW_4cu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EV4ma7OclSdWCsJk_5_aiIvXTgTW_4cu" -O clas_ec.zip && rm -rf ~/cookies.txt`
8. Unzip EC50 Dataset: `unzip clas_ec.zip`
9. Move dataset into ```datasets``` directory : `mv clas_ec datasets`

## Run

# Arguments
* --method : The name of method. One of ['greedy', 'bayesian'].
* --seed : Random seed.
* --sidx : start index of test samples.
* --num_seqs : the number of sequences to attack start from sidx (test_samples[sidx : sidx + num_seqs] is attacked).
* --working_folder : working folder. 
* --block_size : the block size m.
* --max_patience : max patience N_post.

# Baseline (TextFooler)
To reproduce results of the baseline method in table 5, run following codes.
```bash
python attack_codes/attack.py classification --method greedy --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level0
```

```bash
python attack_codes/attack.py classification --method greedy --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level1
```

```bash
python attack_codes/attack.py classification --method greedy --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level2
```

# Our Method
To reproduce results of our method in table 5, run following codes.
```bash
python attack_codes/attack.py classification --method bayesian --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level0 --block_size 20 --max_patience 50
```

```bash
python attack_codes/attack.py classification --method bayesian --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level1 --block_size 20 --max_patience 50
```

```bash
python attack_codes/attack.py classification --method bayesian --seed 0 --sidx 0 --num_seqs 500 --working_folder datasets/clas_ec/clas_ec_ec50_level2 --block_size 20 --max_patience 50
```

# Attack on Protein Classification Dataset, EC50.

## Dependencies
* For evaluation: `pytorch` `fastai` `fire` 
* For bayesian optimization: `botorch` `dppy`

## Setup
1. Create conda environment: `conda create -n protein_atk python=3.9.7` and `conda activate protein_atk`
2. Install pytorch: `conda install pytorch -c pytorch`
3. Install fastai: `conda install -c fastai fastai=1.0.52`
pip install spacy 추가.
4. Install fire: `conda install fire -c conda-forge`
5. Install botorch: `pip install botorch`
6. Install dppy: `pip install dppy`
7. Install sklearn: `pip install sklearn`
7. Download EC50 Dataset: `wget --load-cookies ~/cookies.txt "https://docs.google.com/uc?export=download|confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download|id=1Y3QQpWZ9_fwlXHQTJBNKnOtwVOvCZLib' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')|id=1Y3QQpWZ9_fwlXHQTJBNKnOtwVOvCZLib" -O clas_ec.zip || rm -rf ~/cookies.txt`


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
* --fit-iter : `3` for default setting. (the number of update steps in GP parameter fitting)

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

<table class="center">
  <tr>
    <td> </td>
    <td colspan="3">level 0</td>
    <td colspan="3">level 1</td>
    <td colspan="3">level 2</td>
  </tr>
    <td> Method </td>
    <td> ASR (%) </td>
    <td> MR (%) </td>
    <td> Qrs </td>
    <td> ASR (%) </td>
    <td> MR (%) </td>
    <td> Qrs  </td>
    <td> ASR (%) </td>
    <td> MR (%) </td>
    <td> Qrs </td>
  <tr>
    <td> TF </td>
    <td> 83.8 </td>
    <td> 3.2 </td>
    <td> 619 </td>
    <td> 85.8 </td>
    <td> 3.0 </td>
    <td> 584 </td>
    <td> 89.6 </td>
    <td> 2.5 </td>
    <td> 538 </td>
  </tr>
<tr>
    <td> BBA </td>
    <td> <b>99.8</b> </td>
    <td> <b>2.9</b> </td>
    <td> <b>285</b> </td>
    <td> <b>99.8</b> </td>
    <td> <b>2.3</b> </td>
    <td> <b>293</b> </td>
    <td> <b>100.0</b> </td>
    <td> <b>2.0</b> </td>
    <td> <b>231</b> </td>
  </tr>
</table>

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

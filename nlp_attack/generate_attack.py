import os
import argparse
from collections import defaultdict
import numpy as np


wordnet = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 50",
]

embedding = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
]

hownet = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 100",
]

bert_attack = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-bert-attack --model bert-base-uncased-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type bert-attack --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-bert-attack --model lstm-ag-news --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type bert-attack --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-bert-attack --model xlnet-base-cased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type bert-attack --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-bert-attack --model bert-base-uncased-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type bert-attack --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-bert-attack --model lstm-mr --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type bert-attack --max-patience 20",
]

imdb = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-imdb --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 100",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-imdb --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-imdb --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 50",
]

nli = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet-pre --model bert-base-uncased-mnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 150",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding-pre --model bert-base-uncased-mnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 50",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet-pre --model bert-base-uncased-mnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 150",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet-pre --model bert-base-uncased-qnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 150",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding-pre --model bert-base-uncased-qnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 50",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet-pre --model bert-base-uncased-qnli --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 150",
]

yelp = [
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-wordnet --model bert-base-uncased-yelp --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pwws --max-patience 200",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-embedding --model bert-base-uncased-yelp --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type textfooler --max-patience 20",
    "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --recipe bayesattack-hownet --model bert-base-uncased-yelp --num-examples 500 --sidx 0 --pkl-dir RESULTS --post-opt v3 --use-sod --dpp-type dpp_posterior --max-budget-key-type pso --max-patience 50",
]

#cmds = wordnet + embedding + hownet
cmds = bert_attack


def get_job_name(cmd):
    parts = cmd.split()
    model = parts[11]
    recipe = parts[9]

    return f"{model}_{recipe}"


def main(args):
    template = open(args.template_path, "r").read()

    f = open(f"{args.output_dir}/run.sh", "w")
    f.write("#!/bin/bash\n\n")

    for idx, cmd in enumerate(cmds):
        job_name = get_job_name(cmd)
        g = open(job_name+".sh", "w")
        g.write(template.format(partition=args.partition, workspace=args.workspace, job_name=job_name, cmd=cmd))
        g.close()

        f.write(f"sbatch {job_name}.sh\n")

    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_path", type=str, default="/root/new_attack/DiscreteBlockBayesAttack/nlp_attack/template_sbatch.txt")
    parser.add_argument("--output_dir", type=str, default="/root/new_attack/DiscreteBlockBayesAttack/nlp_attack/")
    parser.add_argument("--partition", type=str, default="applied")
    parser.add_argument("--workspace", type=str, default="/root/new_attack/DiscreteBlockBayesAttack/nlp_attack")

    args = parser.parse_args()
    main(args)

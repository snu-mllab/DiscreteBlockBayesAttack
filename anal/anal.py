from EXP_INFOS import METHODS_INFO
import textattack
from systemize import read_pkl, write_pkl
from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import sys
use = UniversalSentenceEncoder()


def summary(root, model, dataset, key, num_examples):
    success_result = []
    ct = 0
    log_manager = textattack.loggers.AttackLogManager()
    for i in range(num_examples):
        try:
            result = read_pkl(f'{root}/{model}-{dataset}/{key}/{i}.pkl')
            log_manager.results.append(result)
            success_result.append(i)
        except:
            pass
    results = log_manager.results
    return results, success_result, log_manager

def write_text(str_,path):
    f = open(f'{path}', "w")
    f.write(str_)
    f.close()

def lm2eval(log_manager):
    (asr, mr, nq, perp, use, nql, bv) = log_manager.log_summary()
    return asr, mr, nq

def results2avgtime(results):
    tl = []
    for r in results:
        if not isinstance(r,SkippedAttackResult):
            tl.append(r.perturbed_result.elapsed_time)
    print(len(tl))
    return sum(tl) / len(tl)

def write_time_csv(results, max_time, path, mqrs_rate=1.0, use_mb=False, mb=[]):
    tl = []
    tot_num = 0        
    
    if not use_mb:
        mb = [None for _ in range(len(results))]
    for i, r in enumerate(results):
        if not isinstance(r,SkippedAttackResult):
            if use_mb:
                max_budget = mb[i]
            else:
                max_budget = r.perturbed_result.query_budget
                mb[i] = max_budget
            tot_num += 1
        if isinstance(r,SuccessfulAttackResult):
            #print(i)
            #print(r.perturbed_result.elapsed_time)
            #print(r.perturbed_result.num_queries)
            #print(r.perturbed_result.query_budget)
            if r.perturbed_result.num_queries <= mqrs_rate*max_budget:  
                tl.append(r.perturbed_result.elapsed_time)
        
    
    str_ = 'time, asr'
    for tt in range(0,max_time,1):
        num_suc = len([t for t in tl if t <= tt/2])
        asr = num_suc / tot_num
        str_ += f'\n{tt/2}, {asr}'
    write_text(str_, path)
    return mb
    

def fFF():
    for SS, SM in [('wordnet','pwws'),('embedding','textfooler')]:
        MAIN_KEY = f'{SM}_0'
        #MAIN_KEY = f'{SM}_0_product'
        for MODEL, DATASET in [('xlnet-large-cased','yelp')]:
            NUM = 500
            ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP'
            print("DATASET ",DATASET, " MODEL", MODEL)
            print()
            KEY = f'bayesattack-{SS}_40_4_1_50_v3_True_dpp_posterior_5_1_{SM}_0'
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            print("len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            at = results2avgtime(results)
            print(asr, mr, nq, at)
            for mqrs in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                mb = write_time_csv(results, 500, f'./FFFF/{DATASET}-{KEY}_{mqrs}.csv', mqrs_rate=mqrs,use_mb=False,mb= [])
            
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, MAIN_KEY, NUM)
            print("len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            at = results2avgtime(results)
            print(asr, mr, nq, at)
            for mqrs in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                write_time_csv(results, 500, f'./FFFF/{DATASET}-{MAIN_KEY}_{mqrs}.csv', mqrs_rate=mqrs, use_mb=True, mb=mb)
def main_table():
    import os
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP'
    
    
    for MODEL, DATASET in [('bert-base-uncased','imdb'),('bert-base-uncased','yelp'),('bert-base-uncased','mr'),('bert-base-uncased','ag-news'),('lstm','mr'),('lstm','ag-news')]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            print(KEY, asr, mr, nq)
    for MODEL, DATASET in [('bert-large-uncased','imdb'),('bert-large-uncased','yelp'),('bert-large-uncased','mr'),('xlnet-large-cased','imdb'),('xlnet-large-cased','yelp'),('xlnet-large-cased','mr'),('xlnet-base-cased','imdb'),('xlnet-base-cased','yelp'),('xlnet-base-cased','mr'),]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            print(KEY, asr, mr, nq)

main_table()

#fFF()
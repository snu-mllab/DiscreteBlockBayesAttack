from EXP_INFOS import METHODS_INFO
import textattack
from systemize import read_pkl, write_pkl
from textattack.attack_results import FailedAttackResult, SkippedAttackResult, SuccessfulAttackResult
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
import sys
use = UniversalSentenceEncoder()
import os
import numpy as np


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

def lm2eval_f(results):
    fnqs = []
    for i, r in enumerate(results):
        if not isinstance(r,SkippedAttackResult):
            rr = r.perturbed_result
            att_logs = rr.attack_logs
            if isinstance(r,SuccessfulAttackResult):
                evalY = att_logs[2].view(-1)
                for j, y in enumerate(evalY):
                    if y>=0:break
                
                if type(att_logs[3]) == type(None):
                    fnq = j + 1
                else:
                    fnq = j + 2.5
            else:
                fnq = rr.num_queries
            fnqs.append(fnq)
    print(sum(fnqs)/len(fnqs))
                
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
        ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
        #MAIN_KEY = f'{SM}_0'
        MAIN_KEY = f'{SM}_0_product'
        for MODEL, DATASET, pwwsmp in [('xlnet-large-cased','yelp',100),('xlnet-large-cased','imdb',100),('bert-large-uncased','yelp',200),('bert-large-uncased','imdb',150)]:
            NUM = 500
            print("DATASET ",DATASET, " MODEL", MODEL)
            print()
            
            if SM == 'pwws':
                
                KEY = f'bayesattack-{SS}_40_4_1_{pwwsmp}_v3_True_dpp_posterior_5_3_{SM}_0'
            else:
                KEY = f'bayesattack-{SS}_40_4_1_50_v3_True_dpp_posterior_5_3_{SM}_0'
                
            
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            print("len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            at = results2avgtime(results)
            print(asr, mr, nq, at)
            for mqrs in [1.0]:
                mb = write_time_csv(results, 500, f'./GGGG/{MODEL}-{DATASET}-ours-{SM}.csv', mqrs_rate=mqrs,use_mb=False,mb= [])
            
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, MAIN_KEY, NUM)
            print("len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            at = results2avgtime(results)
            print(asr, mr, nq, at)
            for mqrs in [1.0]:
                write_time_csv(results, 500, f'./GGGG/{MODEL}-{DATASET}-{MAIN_KEY}.csv', mqrs_rate=mqrs, use_mb=True, mb=mb)
def main_table():
    import os, sys
    NUM = 500
    #ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3_add'
    
    
    for MODEL, DATASET in [('bert-base-uncased','imdb'),('bert-base-uncased','yelp')]:
    #for MODEL, DATASET in [('lstm','mr')]:
    #for MODEL, DATASET in [('bert-base-uncased','mnli'),('bert-base-uncased','qnli')]:
    #for MODEL, DATASET in [('bert-base-uncased','ag-news'),('lstm','ag-news'),('xlnet-base-cased','mr'),('bert-base-uncased','mr'),('lstm','mr'),('bert-base-uncased','imdb'),('bert-base-uncased','yelp')]:
        
    #for MODEL, DATASET in [('bert-base-uncased','ag-news'),('lstm','ag-news'),('xlnet-base-cased','mr'),('bert-base-uncased','mr'),('lstm','mr'),('bert-base-uncased','imdb'),('bert-base-uncased','yelp')]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        subdirs = [p for p in subdirs if '_5_3_' in p]
        subdirs.sort()
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            else:
                asr, mr, nq = lm2eval(log_manager)
                print(KEY, asr, mr, nq)
    sys.exit()
    for MODEL, DATASET in [('bert-large-uncased','imdb'),('bert-large-uncased','yelp'),('bert-large-uncased','mr'),('xlnet-large-cased','imdb'),('xlnet-large-cased','yelp'),('xlnet-large-cased','mr'),('xlnet-base-cased','imdb'),('xlnet-base-cased','yelp'),('xlnet-base-cased','mr'),]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        #subdirs = [p for p in subdirs if '_5_3_' in p]
        subdirs.sort()
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            print(KEY, asr, mr, nq)

def table4():
    import os, sys
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/TABLE4'
    for MODEL, DATASET in [('bert-base-uncased','ag-news'),('lstm','ag-news'),('bert-base-uncased','mr'),('lstm','mr')]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        subdirs.sort()
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            asr, mr, nq = lm2eval(log_manager)
            print(KEY, asr, mr, nq)
    
def fig3():
    import os, sys
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/FIG3'
    INFOs = [
        ('bert-base-uncased','imdb','bayesattack-embedding_40_4_1_200_v3_True_dpp_posterior_5_3_textfooler_0'),
        ('bert-base-uncased','imdb','bayesattack-wordnet_40_4_1_200_v3_True_dpp_posterior_5_3_pwws_0')
        ]
    ct = 0
    for MODEL, DATASET, KEY in INFOs:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        print()
        print(MODEL, DATASET)
        results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
        
        str_ = 'modif,Qrs\n'
        for mp in range(1,201,4):
            modifl, qrsl = [], []
            for r in results:
                rr = r.perturbed_result
                if isinstance(r,SkippedAttackResult):
                    continue
                att_logs = rr.attack_logs
                if isinstance(r,SuccessfulAttackResult):
                    expl = att_logs[4]
                    assert type(None) != type(expl), "something wrong"
                    for modif, patience, qrs in expl:
                        if 200 - patience >= mp:
                            break
                    modifl.append(modif)
                    qrsl.append(qrs)
                elif isinstance(r,FailedAttackResult):
                    qrsl.append(r.perturbed_result.num_queries)
            
            m = (sum(modifl)/len(modifl) * 100).item()
            q = sum(qrsl)/len(qrsl)
            str_ += f'{m},{q}\n'
        if ct == 0:
            write_text(str_, 'mp_trav_tf.csv')
            ct += 1
        else:
            write_text(str_, 'mp_trav_pwws.csv')
            
def divide():
    import os, sys
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3_add_divide'
    
    for MODEL, DATASET in [('bert-base-uncased','ag-news'),('lstm','ag-news'),('xlnet-base-cased','mr'),('bert-base-uncased','mr'),('lstm','mr'),('bert-base-uncased','imdb'),('bert-base-uncased','yelp')]:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        print()
        print(MODEL, DATASET)
        subdirs = [p for p in subdirs if '_5_3_' in p]
        subdirs.sort()
        for KEY in subdirs:
            results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
            if len(results)!=500:
                print(KEY, "len results", len(results))
            else:
                asr, mr, nq = lm2eval(log_manager)
                print(KEY, asr, mr, nq)

def table4_cal():
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
    INFOs = [
        ('bert-base-uncased','ag-news','bayesattack-wordnet_40_4_1_50_v3_True_dpp_posterior_5_3_pwws_0'),
        ('lstm','ag-news','bayesattack-wordnet_40_4_1_50_v3_True_dpp_posterior_5_3_pwws_0'),
        ('bert-base-uncased','mr','bayesattack-wordnet_40_4_1_100_v3_True_dpp_posterior_5_3_pwws_0'),
        ('lstm','mr','bayesattack-wordnet_40_4_1_50_v3_True_dpp_posterior_5_3_pwws_0')
    ]
    for MODEL, DATASET, KEY in INFOs:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        print()
        print(MODEL, DATASET)
        results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
        if len(results)!=500:
            print(KEY, "len results", len(results))
        else:
            #fasr, fmr, fnq = lm2eval_f(log_manager)
            lm2eval_f(results)
            asr, mr, nq = lm2eval(log_manager)
            print(KEY, asr, mr, nq)

def table2_anal():
    NUM = 500
    ROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
    bROOT = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP'
    
    INFOs = [
        (('bert-base-uncased','ag-news','bayesattack-wordnet_40_4_1_50_v3_True_dpp_posterior_5_3_pwws_0'),
        ('bert-base-uncased','ag-news','pwws_0')),
        (('lstm','ag-news','bayesattack-wordnet_40_4_1_50_v3_True_dpp_posterior_5_3_pwws_0'),
        ('lstm','ag-news','pwws_0'))
    ]
    for (MODEL, DATASET, KEY), (bMODEL, bDATASET, bKEY) in INFOs:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        print()
        print(MODEL, DATASET)
        results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
        bresults, bsuccess_result, blog_manager = summary(bROOT, bMODEL, bDATASET, bKEY, NUM)
        
        results_selected = [r for i, r in enumerate(results) if isinstance(bresults[i],SuccessfulAttackResult) and isinstance(results[i],SuccessfulAttackResult)]
        bresults_selected = [r for i, r in enumerate(bresults) if isinstance(bresults[i],SuccessfulAttackResult) and isinstance(results[i],SuccessfulAttackResult)]
        log_manager.results = results_selected
        blog_manager.results = bresults_selected
        
        print(len([r for i, r in enumerate(results) if isinstance(bresults[i],SuccessfulAttackResult)]))
        print(len([r for i, r in enumerate(results) if isinstance(results[i],SuccessfulAttackResult)]))
        print(len(results_selected))
        lm2eval_f(results)
        asr, mr, nq = lm2eval(log_manager)
        print(KEY, asr, mr, nq)
        asr, mr, nq = lm2eval(blog_manager)
        print(bKEY, asr, mr, nq)

def qrsplots():
    R = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
    R2 = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3_add'
    R3 = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3_add_divide'
    NUM = 500
    INFOs = [
        ('pwws',R2,'bert-base-uncased','yelp','bayesattack-wordnet_40_4_1_200_v3_True_dpp_posterior_5_3_pwws_0',5000),
        ('textfooler',R,'bert-base-uncased','yelp','bayesattack-embedding_40_4_1_20_v3_True_dpp_posterior_5_3_textfooler_0',2000),
        ('pso',R3,'bert-base-uncased','yelp','bayesattack-hownet_40_4_1_50_v3_True_dpp_posterior_5_3_pso_0',5000),
        ('lsh',R2,'bert-base-uncased','yelp','bayesattack-hownet_40_4_1_100_v3_True_dpp_posterior_5_3_lsh_0',3000),
    ]

    INFOs2 = [
        ('pwws',R,'bert-base-uncased','imdb','bayesattack-wordnet_40_4_1_100_v3_True_dpp_posterior_5_3_pwws_0',5000),
        ('textfooler',R,'bert-base-uncased','imdb','bayesattack-embedding_40_4_1_20_v3_True_dpp_posterior_5_3_textfooler_0',5000),
        ('pso',R3,'bert-base-uncased','imdb','bayesattack-hownet_40_4_1_50_v3_True_dpp_posterior_5_3_pso_0',5000),
        ('lsh',R2,'bert-base-uncased','imdb','bayesattack-hownet_40_4_1_100_v3_True_dpp_posterior_5_3_lsh_0',5000),
    ]
    
    INFOs3 = [
        ('pwws',R2,'bert-base-uncased','mnli','bayesattack-wordnet-pre_40_4_1_150_v3_True_dpp_posterior_5_3_pwws_0',400),
        ('textfooler',R2,'bert-base-uncased','mnli','bayesattack-embedding-pre_40_4_1_50_v3_True_dpp_posterior_5_3_textfooler_0',400),
        ('pso',R2,'bert-base-uncased','mnli','bayesattack-hownet-pre_40_4_1_150_v3_True_dpp_posterior_5_3_pso_0',1000),
    ]
    INFOs4 = [
        ('pwws',R,'bert-base-uncased','mr','bayesattack-wordnet_40_4_1_100_v3_True_dpp_posterior_5_3_pwws_0',500),
        ('textfooler',R,'bert-base-uncased','mr','bayesattack-embedding_40_4_1_20_v3_True_dpp_posterior_5_3_textfooler_0',500),
        ('pso',R3,'bert-base-uncased','mr','bayesattack-hownet_40_4_1_100_v3_True_dpp_posterior_5_3_pso_0',1000),
    ]
    for mbkt, ROOT, MODEL, DATASET, KEY, MB in INFOs+INFOs2+INFOs3+INFOs4:
        cur_dir = ROOT + f'/{MODEL}-{DATASET}/'
        print()
        print(MODEL, DATASET)
        results, success_result, log_manager = summary(ROOT, MODEL, DATASET, KEY, NUM)
        (asr, mr, nq, perp, use, num_queries, bv) = log_manager.log_summary()
        save_key = f'ours_final/{DATASET}-ours-{mbkt}.csv'
        write_csv_nql(num_queries,bv,save_key,MB)

def write_csv_nql(num_queries,bv,path,MB=5000):
    mq = min(np.max(num_queries),MB)
    if mq >= 2000: step_size = 20
    elif mq <= 500: step_size = 5
    else: step_size = 10
    str_ = 'qrs,ours\n'
    for n in range(0,mq+step_size,step_size):
        asr = np.sum((num_queries<=n)*bv)/len(num_queries)*100
        str_ += f'{n},{asr}\n'
    write_text(str_,path)

def model_info():
    R = f'/home/deokjae/ICML2022_BBA/nlp_attack/EXP_v3'
    NUM = 500
    for MODEL, DATASET in [('bert-base-uncased','ag-news'),('lstm','ag-news'),('xlnet-base-cased','mr'),('bert-base-uncased','mr'),('lstm','mr'),('bert-base-uncased','imdb'),('bert-base-uncased','yelp'), ('bert-large-uncased','imdb'),('bert-large-uncased','yelp'),('bert-large-uncased','mr'),('xlnet-large-cased','imdb'),('xlnet-large-cased','yelp'),('xlnet-large-cased','mr'),('xlnet-base-cased','imdb'),('xlnet-base-cased','yelp'),('xlnet-base-cased','mr')]:
        cur_dir = R + f'/{MODEL}-{DATASET}/'
        subdirs = os.listdir(cur_dir)
        KEY = subdirs[0]
        results, success_result, log_manager = summary(R, MODEL, DATASET, KEY, NUM)
        assert len(results) == 500, "something wrong"
        acc, len_seq= get_model_info(results)
        print(f'{DATASET}&{MODEL}&{len_seq:.1f}&{acc*100:.1f}\\\\')

def get_model_info(results):
    acc = 0
    len_seq = 0
    for r in results:
        if not isinstance(r,SkippedAttackResult):
            acc += 1
        tt = r.perturbed_result.attacked_text
        len_seq += len(tt.words)
    acc = acc / 500
    len_seq = len_seq / 500
    return acc, len_seq
    

#main_table()
#divide()
#table4()
#table4_cal()
#table2_anal()
#fig3()
fFF()
#qrsplots()
#model_info()
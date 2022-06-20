from slurm_launcher.sbatch_launcher import launch_tasks
import time
def large_baselines():
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir EXP --dataset-split test --product-space" # do not include 'python' in your file name
    PARAM_DICT = {
        "--recipe" : ['pwws','textfooler'],
        "--model" : [
            'pretrained_models/xlnet-large-cased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
            'pretrained_models/xlnet-large-cased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
            'pretrained_models/xlnet-large-cased-imdb/best_model/ --dataset-from-huggingface imdb',
            
            'pretrained_models/xlnet-base-cased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
            'pretrained_models/xlnet-base-cased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
            'pretrained_models/xlnet-base-cased-imdb/best_model/ --dataset-from-huggingface imdb',
            
            'pretrained_models/bert-large-uncased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
            'pretrained_models/bert-large-uncased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
            'pretrained_models/bert-large-uncased-imdb/best_model/ --dataset-from-huggingface imdb',
                     ],
        "--sidx" : [0,250],
        #"--sidx" : list(range(0,500,25)),
        "--num-examples" : [250],
        }
    launch_tasks(
        param_option=1,
        base_cmd=PYTHON_FILE,
        param_dict=PARAM_DICT,
        partition='rtx3090',
        exclude='radish,quiznos',
        qos='normal',
        timeout='INFINITE',
        job_name='lbtime',
    )

def large_ours():
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir EXP_v3 --dataset-split test --use-sod" # do not include 'python' in your file name
    for recipe, mbkt in [('wordnet','pwws')]:#,('embedding','textfooler')]:
        PARAM_DICT = {
            "--recipe" : [f'bayesattack-{recipe}'],
            "--model" : [
                #'pretrained_models/xlnet-large-cased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
                #'pretrained_models/xlnet-large-cased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
                #'pretrained_models/xlnet-large-cased-imdb/best_model/ --dataset-from-huggingface imdb',
                
            #'pretrained_models/xlnet-base-cased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
            #'pretrained_models/xlnet-base-cased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
            #'pretrained_models/xlnet-base-cased-imdb/best_model/ --dataset-from-huggingface imdb',
            
            'pretrained_models/bert-large-uncased-mr/best_model/ --dataset-from-huggingface rotten_tomatoes',
            'pretrained_models/bert-large-uncased-yelp/best_model/ --dataset-from-huggingface yelp_polarity',
            'pretrained_models/bert-large-uncased-imdb/best_model/ --dataset-from-huggingface imdb',
                        ],
            "--sidx" : [0,250],
            "--num-examples" : [250],
            "--post-opt" : ['v3'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : [mbkt],
            "--max-patience" : [150,200],
            "--fit-iter" : [3],
            }
        launch_tasks(
            param_option=1,
            base_cmd=PYTHON_FILE,
            param_dict=PARAM_DICT,
            partition='rtx3090',
            exclude='radish,geoffrey,icecream',
            qos='normal',
            timeout='INFINITE',
            job_name='lotimev3',
        )

def main_table():
    
    '''
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir BAE --product-space" # do not include 'python' in your file name
    PARAM_DICT = {
        #"--recipe" : ['pwws','textfooler','pso','bae'],
        "--recipe" : ['bae'],
        "--model" : ['bert-base-uncased-yelp','bert-base-uncased-ag-news','bert-base-uncased-mr','bert-base-uncased-imdb','lstm-mr','lstm-ag-news','xlnet-base-cased-mr'],
        "--sidx" : [50,150,250,350,450],
        "--num-examples" : [50],
        }
    launch_tasks(
        param_option=1,
        base_cmd=PYTHON_FILE,
        param_dict=PARAM_DICT,
        partition='dept,titan,rtx2080,rtx3090',
        exclude='radish,geoffrey,icecream',
        qos='normal',
        timeout='INFINITE',
        job_name='main33',
    )
    
    
    '''
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir EXP_v3_add_divide --use-sod" # do not include 'python' in your file name
    large_models = ['bert-base-uncased-imdb','bert-base-uncased-yelp']
    small_models = ['bert-base-uncased-ag-news','bert-base-uncased-mr','lstm-mr','lstm-ag-news','xlnet-base-cased-mr']
    #for recipe, mbkt in [('wordnet','pwws'),('embedding','textfooler'),('hownet','pso'),('bae','bae'),('wordnet','lsh'),('embedding','lsh'),('hownet','lsh')]:
    #for recipe, mbkt in [('wordnet','pwws'),('hownet','pso'),('embedding','lsh'),('hownet','lsh')]:
    #for recipe, mbkt in [('wordnet','lsh'),('embedding','lsh'),('hownet','lsh')]:
    #for recipe, mbkt in [('embedding','textfooler')]:
    for recipe, mbkt in [('hownet','pso')]:
        for size in ['large','small']:
            if size == 'large': 
                models = large_models
                if mbkt in ['pwws','pso']:
                    #mp = [200,250,300]
                    mp = [50]
                else:
                    mp = [100]
                fit_iters = [3]
            else: 
                if mbkt == 'lsh': continue
                models = small_models
                fit_iters = [3]
                if mbkt in ['pwws','pso']:
                    mp = [50,100]
                else:
                    mp = [20,50]
            
            
            PARAM_DICT = {
            "--recipe" : [f'bayesattack-{recipe}'],
            "--model" : models,
            "--sidx" : [0,250],
            "--num-examples" : [250],
            "--post-opt" : ['v3'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : [mbkt],
            "--max-patience" : mp,
            "--fit-iter" : fit_iters,
            }
            launch_tasks(
                param_option=1,
                base_cmd=PYTHON_FILE,
                param_dict=PARAM_DICT,
                partition='rtx3090',
                exclude='radish,geoffrey,icecream',
                qos='normal',
                timeout='INFINITE',
                job_name='mainv3divide',
            )
    
def add_table():
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir EXP_new --use-sod" # do not include 'python' in your file name
    for recipe, mbkt in [('embedding','textfooler')]:
            PARAM_DICT = {
            "--recipe" : [f'bayesattack-{recipe}'],
            "--model" : ['lstm-mr'],
            "--sidx" : [0],
            "--num-examples" : [500],
            "--post-opt" : ['v3','v4'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : [mbkt],
            "--max-patience" : [6,10,15],
            "--fit-iter" :  [3],
            }
            launch_tasks(
                param_option=1,
                base_cmd=PYTHON_FILE,
                param_dict=PARAM_DICT,
                partition='dept,titan,rtx2080',
                exclude='radish',
                qos='normal',
                timeout='INFINITE',
                job_name='main',
            )
    

def nli_table():
    
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir EXP_v3_add --use-sod" 
    models = ['bert-base-uncased-mnli','bert-base-uncased-qnli']

    for m in models:
        for recipe, mbkt in [('wordnet-pre','lsh'),('embedding-pre','lsh'),('hownet-pre','lsh')]:
        #for recipe, mbkt in [('wordnet-pre','pwws'),('embedding-pre','textfooler'),('hownet-pre','pso'),('wordnet-pre','lsh'),('embedding-pre','lsh'),('hownet-pre','lsh')]:
            fit_iters = [3]
            if mbkt in ['pwws','pso']:
                mp = [100,150,200,250,300]
            else:
                mp = [20,50,100,150,200]
            
            if 'qnli' in m and mbkt == 'lsh':
                continue
            
            
            PARAM_DICT = {
            "--recipe" : [f'bayesattack-{recipe}'],
            "--model" : [m],
            "--sidx" : [0],
            "--num-examples" : [500],
            "--post-opt" : ['v3'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : [mbkt],
            "--max-patience" : mp,
            "--fit-iter" : fit_iters,
            }
            launch_tasks(
                param_option=1,
                base_cmd=PYTHON_FILE,
                param_dict=PARAM_DICT,
                partition='dept,titan,rtx2080,rtx3090',
                exclude='radish,geoffrey,icecream',
                qos='normal',
                timeout='INFINITE',
                job_name='mainv33',
            )

def table4():
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir TABLE4 --use-sod" 
    models = ['bert-base-uncased-ag-news','lstm-ag-news','bert-base-uncased-mr','lstm-mr']
    PARAM_DICT = {
            "--recipe" : ['bayesattack-wordnet'],
            "--model" : models,
            "--sidx" : list(range(0,500,250)),
            "--num-examples" : [250],
            "--post-opt" : ['none'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : ['pwws'],
            "--max-patience" : [0],
            "--fit-iter" : [3],
            "--batch-size" : [4],
            "--update-step" : [1],
            }
    launch_tasks(
                param_option=1,
                base_cmd=PYTHON_FILE,
                param_dict=PARAM_DICT,
                partition='dept,titan,rtx2080,rtx3090',
                exclude='radish,geoffrey,icecream',
                qos='normal',
                timeout='INFINITE',
                job_name='mainv33',
            )

def fig3():
    PYTHON_FILE = "textattack attack --silent --shuffle --shuffle-seed 0 --random-seed 0 --pkl-dir FIG3 --use-sod" 
    models = ['bert-base-uncased-imdb']
    PARAM_DICT = {
            "--recipe" : ['bayesattack-wordnet'],
            "--model" : models,
            "--sidx" : list(range(0,500,50)),
            "--num-examples" : [50],
            "--post-opt" : ['v3'],
            "--dpp-type" : ['dpp_posterior'],
            "--max-budget-key-type" : ['pwws'],
            "--max-patience" : [200],
            "--fit-iter" : [3],
            }
    launch_tasks(
                param_option=1,
                base_cmd=PYTHON_FILE,
                param_dict=PARAM_DICT,
                partition='dept,titan,rtx2080,rtx3090',
                exclude='radish,geoffrey,icecream',
                qos='normal',
                timeout='INFINITE',
                job_name='fig3',
            )
if __name__=='__main__':
    fig3()
    #table4()
    #add_table()
    #main_table()
    #nli_table()
    #large_baselines()
    #large_ours()
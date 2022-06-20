from slurm_launcher.sbatch_launcher import launch_tasks


def main_table():
    PYTHON_FILE = "python attack_codes/attack.py classification --eval 1" # do not include 'python' in your file name
    PARAM_DICT = {
        "--method" : ["bayesian","greedy"],
        "--seed" : [0],
        "--sidx" : [0],
        "--num_seqs" : [500],
        "--working_folder" : ["datasets/clas_ec/clas_ec_ec50_level0", "datasets/clas_ec/clas_ec_ec50_level1", "datasets/clas_ec/clas_ec_ec50_level2"],
        "--block_size" : [20],
        "--max_patience" : [50],
        }
    launch_tasks(
        param_option=1,
        base_cmd=PYTHON_FILE,
        param_dict=PARAM_DICT,
        partition='dept,titan,rtx2080,rtx3090',
        exclude='',
        qos='normal',
        timeout='INFINITE',
        job_name='ptmain_eval',
    )
    
def block_size_exp():
    PYTHON_FILE = "python attack_codes/attack.py classification --eval 1" # do not include 'python' in your file name
    PARAM_DICT = {
        "--method" : ["bayesian"],
        "--seed" : [0],
        "--sidx" : [0],
        "--num_seqs" : [500],
        "--working_folder" : ["datasets/clas_ec/clas_ec_ec50_level1"],
        "--block_size" : [2,5,10,20,40,80],
        "--max_patience" : [50],
        }
    launch_tasks(
        param_option=1,
        base_cmd=PYTHON_FILE,
        param_dict=PARAM_DICT,
        partition='dept,titan,rtx2080,rtx3090',
        exclude='',
        qos='normal',
        timeout='INFINITE',
        job_name='ptsupp_eval',
    )
if __name__=='__main__':
    #main_table()
    block_size_exp()
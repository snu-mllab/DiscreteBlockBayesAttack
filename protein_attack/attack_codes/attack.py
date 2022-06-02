'''code version consistent with version v1 of the fastai library
follow installation instructions at https://github.com/fastai/fastai
'''
import fire
from warnings import filterwarnings
filterwarnings("ignore")
import os, sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
from model_utils import *
from attack_codes.attack_util import get_data_and_learner, load_pretrained_model, fix_seed
from fastai.basic_train import loss_batch
import time
from attack_codes.gen_synonym import get_synonym
from attack_codes.greedy_attack import greedy_attack
from attack_codes.bayesian_attack import bayesian_attack

kwargs_defaults = {
"working_folder":"datasets/clas_ec/clas_ec_ec50_level1", # folder with preprocessed data 
"model_filename_prefix":"model", # filename for saved model
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":400, # embedding size
"nh":1150, # number of hidden units
"nl":3, # number of layers

"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":128, # batch size

"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)

############ For attack ##############
"method": "bayesian",
"block_size": 40,
"max_patience": 20,
"eval": 0,
"seed": 0,
"sidx": 0,
"num_seqs":50,
"save_key":"",
}
debug = True

def get_accuracy(dataset, BBM):
    correct = 0
    t0 = time.time()
    # Original Accuracy
    score_ct = 0
    with torch.no_grad():
        val_losses, ybs = [], []
        for xb,yb in dataset:
            _, y_pred = BBM.get_pred(xb, yb)
            score = BBM.get_score(xb,yb)
            val_losses.append(y_pred)
            ybs.append(yb)
            if score > 0: score_ct += 1
    outs = torch.cat(val_losses, dim=0).cpu().detach()
    ybs = torch.cat(ybs).cpu().detach()
    correct = torch.sum((outs == ybs))
    Orig_Acc = (correct/outs.shape[0]).item()
    print(f"Original Accuracy via get_pred function : {100*Orig_Acc:.2f}%")
    print(f"Original Accuracy via get_score function : {100-score_ct/outs.shape[0]*100:.2f}")
    return Orig_Acc


def evaluate_adv(results, BBM):
    nql, modifl, succl = [], [], []
    for xadv, xb, yb, nq, modif, succ in results:
        if BBM.get_score(xb, yb) >= 0:
            assert succ == -1, "something wrong"
        else:
            if BBM.get_score(xadv, yb) >= 0:
                assert succ == 1, "something wrong"
                modifl.append(modif)
            else:
                assert succ == 0, "something wrong"
            nql.append(nq)
            succl.append(succ)
    asr = '{:.2f}'.format(sum(succl)/len(succl)*100)
    anq = '{:.1f}'.format(sum(nql)/len(nql))
    am = '{:.2f}'.format(sum(modifl)/len(modifl)*100)
    return asr, am, anq

def get_adv_dir(working_folder, kwargs):
    if kwargs['method'] == 'bayesian':
        adv_dir = working_folder/'BAYES{}_{}_{}{}'.format(kwargs['seed'],kwargs['block_size'],kwargs['max_patience'],kwargs['save_key'])
    elif kwargs['method'] == 'greedy':
        adv_dir = working_folder/'GREEDY{}{}'.format(kwargs['seed'],kwargs['save_key'])
    else:
        raise ValueError
    print("------------------------------------------")
    print(adv_dir)
    return adv_dir

class BlackBoxModel():
    def __init__(self, model, cb_handler, syndict):
        self.eval_cache = dict()
        self.initialize_num_queries()
        self.model = model
        self.cb_handler = cb_handler
        self.syndict = syndict

    def initialize_num_queries(self):
        self.clean_cache()
        self.num_queries = 0
    
    def clean_cache(self):
        del self.eval_cache
        self.eval_cache = dict()
    
    def set_query_budget(self, query_budget):
        self.query_budget=query_budget
    
    def set_y(self, y):
        if type(y) == np.ndarray: y = torch.LongTensor(y)
        self.y = y

    def get_pred(self, x):
        '''
            output:
                pred : 1 x nlabel tensor
                y_pred : 1 tensor
        '''
        if type(x) == np.ndarray: x = torch.LongTensor(x)
        with torch.no_grad():
            if x.cpu().detach() in self.eval_cache:
                val_loss = self.eval_cache[x.cpu().detach()]
            else:
                val_loss = loss_batch(self.model, x.cuda(), self.y.cuda(), cb_handler=self.cb_handler)
                self.eval_cache[x.cpu().detach()] = val_loss
            self.num_queries += 1
        return val_loss[0], torch.argmax(val_loss[0]).view(1)
    
    def get_score(self,x):
        if type(x) == np.ndarray: x = torch.LongTensor(x)
        with torch.no_grad():
            pred, _ = self.get_pred(x)
            prob = torch.nn.functional.softmax(pred)
            y_ = self.y.cpu().detach().item()
            prob_del = torch.cat([prob[:,:y_],prob[:,y_+1:]],dim=-1)
            score = (torch.max(prob_del) - prob[0][self.y]).cpu().detach().item()
        return score

def generic_model(clas=True, **kwargs):
    kwargs["clas"]=clas
    for k in kwargs_defaults.keys():
        if(not( k in kwargs.keys()) or kwargs[k] is None):
            kwargs[k]=kwargs_defaults[k]
    
    WORKING_FOLDER = Path(kwargs["working_folder"])    
    ADV_DIR = get_adv_dir(WORKING_FOLDER,kwargs)
    ADV_DIR.mkdir(exist_ok=True)
    fix_seed(seed=kwargs['seed'])

    learn, target_seqs, vocab = get_data_and_learner(WORKING_FOLDER, kwargs, num_seqs=500) # get shuffled data
    cb_handler = CallbackHandler(learn.callbacks, learn.metrics)
    model = load_pretrained_model(learn, kwargs)
    model.eval()

    syndict = get_synonym(vocab)

    BBM = BlackBoxModel(model, cb_handler, syndict)

    results = []
    if kwargs['eval']==1:
        orig_acc = get_accuracy(target_seqs, BBM) # accuracy of whole test dataset

        for key, l_ in syndict.items(): print(key, l_)
        avg_len = 0
        for i in range(kwargs['sidx'],kwargs['sidx'] + kwargs['num_seqs']):
            ADV_PATH = ADV_DIR/'{}.npy'.format(int(i))
            result = np.load(ADV_PATH,allow_pickle=True)
            results.append(result)
            xb_att = result[0]
            avg_len += xb_att.shape[1]
            print_att_info(kwargs['sidx']+i, result[3], result[4], result[5])
        avg_len = avg_len / kwargs['num_seqs']
        print(f"orig acc : {orig_acc}, avg len : {avg_len}")
    else:
        for idx, (xb,yb) in enumerate(target_seqs[int(kwargs['sidx']):int(kwargs['sidx']+kwargs['num_seqs'])]):
            ADV_PATH = ADV_DIR/'{}.npy'.format(int(kwargs['sidx'] + idx))
            try:
                result = np.load(ADV_PATH,allow_pickle=True)
                results.append(result)
                xb_att, xb, yb, num_queries, modif_rate, succ = result
            except:
                if kwargs['method'] == 'greedy': 
                    xb_att, num_queries, modif_rate, succ = greedy_attack(xb, yb, syndict, BBM)
                elif kwargs['method'] == 'bayesian':
                    xb_att, num_queries, modif_rate, succ = bayesian_attack(xb, yb, syndict, BBM, block_size=kwargs['block_size'], max_patience=kwargs['max_patience'])
                result = [xb_att.cpu().detach().numpy(), xb.cpu().detach().numpy(), yb.cpu().detach().numpy(), num_queries, modif_rate, succ]
                np.save(ADV_PATH, result, allow_pickle=True)
                results.append(result)
            print_att_info(kwargs['sidx']+idx, num_queries, modif_rate, succ)
    asr, am, anq = evaluate_adv(results, BBM)
    print(f"Avg. Att. Succ. Rate : {asr}")
    print(f"Avg. Modif. Rate : {am}")
    print(f"Avg. Num Queries : {anq}")
    learn.destroy()
    return [asr, am, anq]


def print_att_info(index, qrs, mr, succ):
    str_ = f'{index}th sample '
    if succ == -1:
        str_ += 'SKIPPED'
    elif succ == 0:
        str_ += 'FAILED\n'
        str_ += f"num queries : {qrs*100:.1f}"
    elif succ == 1:
        str_ += 'SUCCESSED\n'
        str_ += f"num queries : {qrs*100:.1f}, modif rate : {100*mr:.1f}"
    print(str_)
    
class Model(object):
    def generic_model(self, **kwargs):
        return generic_model(**kwargs)

    def languagemodel(self, **kwargs):
        return self.generic_model(clas=False, **kwargs)
    
    def classification(self, **kwargs):
        return self.generic_model(clas=True, **kwargs)
    
if __name__ == '__main__':
    fire.Fire(Model)

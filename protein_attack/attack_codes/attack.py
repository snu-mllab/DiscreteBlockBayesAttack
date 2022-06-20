'''code version consistent with version v1 of the fastai library
follow installation instructions at https://github.com/fastai/fastai
'''
import fire
from warnings import filterwarnings
from  copy import deepcopy
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
"block_size": 20,
"max_patience": 20,
"fit_iter": 3,
"eval": 0,
"seed": 0,
"sidx": 0,
"num_seqs":50,
"save_key":"",
}
debug = True

class Model(object):
    def generic_model(self, **kwargs):
        return generic_model(**kwargs)

    def languagemodel(self, **kwargs):
        return self.generic_model(clas=False, **kwargs)
    
    def classification(self, **kwargs):
        return self.generic_model(clas=True, **kwargs)

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
            print_att_info(kwargs['sidx']+i, result[3], result[4], result[5], result[6])
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
                    xb_att, num_queries, modif_rate, succ, elapsed_time, attack_logs = greedy_attack(xb, yb, syndict, BBM)
                elif kwargs['method'] == 'bayesian':
                    xb_att, num_queries, modif_rate, succ, elapsed_time, attack_logs = bayesian_attack(xb, yb, syndict, BBM, block_size=kwargs['block_size'], max_patience=kwargs['max_patience'], fit_iter=kwargs['fit_iter'])
                result = [xb_att.cpu().detach().numpy(), xb.cpu().detach().numpy(), yb.cpu().detach().numpy(), num_queries, modif_rate, succ, elapsed_time, attack_logs]
                np.save(ADV_PATH, result, allow_pickle=True)
                results.append(result)
            print_att_info(kwargs['sidx']+idx, num_queries, modif_rate, succ, elapsed_time)
    asr, am, anq, at = evaluate_adv(results, BBM)
    print(f"Avg. Att. Succ. Rate : {asr}")
    print(f"Avg. Modif. Rate : {am}")
    print(f"Avg. Num Queries : {anq}")
    print(f"Avg. Elapsed Time : {at}")
    learn.destroy()
    return [asr, am, anq, at]    

def get_accuracy(dataset, BBM):
    correct = 0
    t0 = time.time()
    # Original Accuracy
    score_ct = 0
    with torch.no_grad():
        val_losses, ybs = [], []
        for xb,yb in dataset:
            xb = xb.cpu().detach()
            yb = yb.cpu().detach()
            BBM.set_xy(xb,yb)
            _, y_pred = BBM.get_pred(xb)
            score = BBM.get_score(xb,require_transform=False)
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
    nql, modifl, succl, etl = [], [], [], []
    BBM.set_query_budget(float('inf'))
    for xadv, xb, yb, nq, modif, succ, elapsed_time, attack_logs in results:
        BBM.set_xy(xb, yb)
        if BBM.get_score(xb, require_transform=False) >= 0:
            assert succ == -1, "something wrong"
        else:
            if BBM.get_score(xadv, require_transform=False) >= 0:
                assert succ == 1, "something wrong"
                modifl.append(modif)
            else:
                assert succ == 0, "something wrong"
            nql.append(nq)
            succl.append(succ)
            etl.append(elapsed_time)            
            
    asr = '{:.2f}'.format(sum(succl)/len(succl)*100)
    anq = '{:.1f}'.format(sum(nql)/len(nql))
    am = '{:.2f}'.format(sum(modifl)/len(modifl)*100)
    at = '{:.2f}'.format(sum(etl)/len(etl))
    return asr, am, anq, at

def get_adv_dir(working_folder, kwargs):
    if kwargs['method'] == 'bayesian':
        adv_dir = working_folder/'BAYES{}_{}_{}_{}{}'.format(kwargs['seed'],kwargs['block_size'],kwargs['max_patience'],kwargs['fit_iter'],kwargs['save_key'])
    elif kwargs['method'] == 'greedy':
        adv_dir = working_folder/'GREEDY{}{}'.format(kwargs['seed'],kwargs['save_key'])
    else:
        raise ValueError
    print("------------------------------------------")
    print(adv_dir)
    return adv_dir

def print_att_info(index, qrs, mr, succ, elapsed_time):
    str_ = f'{index}th sample '
    if succ == -1:
        str_ += 'SKIPPED'
    elif succ == 0:
        str_ += 'FAILED\n'
        str_ += f"num queries : {qrs:.1f}"
    elif succ == 1:
        str_ += 'SUCCESSED\n'
        str_ += f"num queries : {qrs:.1f}, modif rate : {100*mr:.1f}"
    str_ += f"\nelapsed_time : {elapsed_time:.2f}"
    print(str_)
    
class BlackBoxModel():
    def __init__(self, model, cb_handler, syndict):
        self.eval_cache = dict()
        self.initialize_num_queries()
        self.model = model
        self.cb_handler = cb_handler
        self.syndict = syndict
        self.query_budget = float('inf')

    def initialize_num_queries(self):
        self.clean_cache()
        self.num_queries = 0
    
    def clean_cache(self):
        del self.eval_cache
        self.eval_cache = dict()
    
    def set_query_budget(self, query_budget):
        self.query_budget=query_budget
    
    def set_xy(self, x0, y):
        if type(x0) == np.ndarray: x0 = torch.LongTensor(x0)
        if type(y) == np.ndarray: y = torch.LongTensor(y)
        self.x0 = x0
        self.y = y
        self.len_seq = x0.numel()
        self.word_substitution_cache = [[] for _ in range(self.len_seq)]
        for ind in range(self.len_seq):
            self.word_substitution_cache[ind] = deepcopy(self.syndict[x0[0][ind].cpu().item()])
        self.n_vertices = [len(w_candids) for w_candids in self.word_substitution_cache]
        self.target_indices = [ind for ind in range(self.len_seq) if self.n_vertices[ind]>1]
    
    def seq2input(self, seq):
        assert type(seq) == torch.Tensor, f"type(seq) is {type(seq)}"
        if len(seq.shape) == 1:
            assert seq.shape[0] == self.len_seq, "indices length should be seq length"
            seq_ = seq
        elif len(seq.shape) == 2:
            assert seq.shape[0] == 1 and seq.shape[1] == self.len_seq, "indices length should be seq length"
            seq_ = seq.view(-1)
        cur_seq = self.x0
        modified_indices = [ct for ct, ind in enumerate(seq_) if ind > 0 and ct in self.target_indices]
        words = [self.word_substitution_cache[ct][int(ind)] for ct, ind in enumerate(seq_) if ind > 0 and ct in self.target_indices]
        new_seq = self.replace_words_at_indices(cur_seq,modified_indices, words)
        return new_seq
    
    def replace_words_at_indices(self, seq, modified_indices, words):
        new_seq = deepcopy(seq)
        for ind, word in zip(modified_indices, words):
            new_seq[0][ind] = word
        return new_seq    

    def get_initial_block_order(self, inds_list):
        index_scores = []
        for inds in inds_list:
            start, end = self.target_indices[inds[0]], self.target_indices[inds[-1]]
            del_seq = deepcopy(self.x0)
            del_seq = torch.cat([del_seq[:,:start],del_seq[:,end+1:]],dim=-1)
            score = self.get_score(del_seq,require_transform=False)
            index_scores.append(score)
        index_scores = np.array(index_scores)
        index_order = (-index_scores).argsort()
        return index_order

    def seq2str(self, seq):
        assert seq.type() == 'torch.LongTensor', f"{seq.type()} should be 'torch.LongTensor'"
        seq_ = seq.view(-1).cpu().detach()
        str_ = ''
        for i in seq_:
            str_ += f'{int(i)},'
        return str_
        
    def get_pred(self, x):
        '''
            output:
                pred : 1 x nlabel tensor
                y_pred : 1 tensor
        '''
        if type(x) == np.ndarray: x = torch.LongTensor(x)
        if x.type() != torch.LongTensor: x = x.long()
        with torch.no_grad():
            x_str = self.seq2str(x)
            if x_str in self.eval_cache:
                val_loss = self.eval_cache[x_str]
            else:
                if self.num_queries >= self.query_budget: return None, None
                val_loss = loss_batch(self.model, x.cuda(), self.y.cuda(), cb_handler=self.cb_handler)
                self.eval_cache[x_str] = val_loss
                self.num_queries += 1
        return val_loss[0], torch.argmax(val_loss[0]).view(1)
    
    def get_score(self,x,require_transform=True):
        if require_transform:
            x_ = self.seq2input(x.view(-1))
        else:
            x_ = x
        with torch.no_grad():
            pred, _ = self.get_pred(x_)
            if type(pred) == type(None): return None
            prob = torch.nn.functional.softmax(pred)
            y_ = self.y.cpu().detach().item()
            prob_del = torch.cat([prob[:,:y_],prob[:,y_+1:]],dim=-1)
            score = (torch.max(prob_del) - prob[0][self.y]).cpu().detach().item()
        return score
    
    def get_scores(self, xs, require_transform=True):
        s = []
        for x in xs:
            s.append(self.get_score(x,require_transform=require_transform))
        return s

if __name__ == '__main__':
    fire.Fire(Model)

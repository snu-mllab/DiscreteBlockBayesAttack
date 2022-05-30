from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.metrics import fbeta_score
from model_utils import *

def get_data_and_learner(WORKING_FOLDER, kwargs, num_seqs=500):
    write_log_header(WORKING_FOLDER,kwargs)
    clas = kwargs['clas']

    # Load preprocessed data
    tok = np.load(WORKING_FOLDER/'tok.npy', allow_pickle=True)
    if(clas):
        label = np.load(WORKING_FOLDER/'label.npy')
    
    # dtype issue if all sequences of same length (numpy turns array of lists into matrix)
    # turn matrix into array of python lists
    if tok.dtype is np.dtype("int32"):
        tok_list = np.empty(tok.shape[0], dtype=np.object)
        for i in range(tok.shape[0]):
            tok_list[i] = []
            tok_list[i].extend(tok[i].tolist())
        tok = tok_list
    
    #get train/val/test IDs
    train_IDs_raw = np.load(WORKING_FOLDER/'train_IDs.npy',allow_pickle=True)
    val_IDs_raw = np.load(WORKING_FOLDER/'val_IDs.npy',allow_pickle=True)
    test_IDs_raw = np.load(WORKING_FOLDER/'test_IDs.npy',allow_pickle=True)

    #print(val_IDs_raw[:20])
    #print(test_IDs_raw[:20])

    train_IDs = train_IDs_raw 
    val_IDs = val_IDs_raw
    test_IDs = test_IDs_raw

    tok_itos = np.load(WORKING_FOLDER/'tok_itos.npy',allow_pickle=True)
    
    PRETRAINED_FOLDER = WORKING_FOLDER
    if((PRETRAINED_FOLDER/'tok_itos.npy').exists()):
        tok_itos_pretrained = np.load(PRETRAINED_FOLDER/'tok_itos.npy')
        #if(len(tok_itos)!=len(tok_itos_pretrained) or np.all(tok_itos_pretrained==tok_itos) is False):#nothing to do
        if(len(tok_itos)!=len(tok_itos_pretrained) or ~np.all(tok_itos_pretrained==tok_itos)):
            assert(len(tok_itos)<=len(tok_itos_pretrained)) #otherwise the vocab size does not work out
            print("tok_itos does not match- remapping...")
            print("tok_itos_pretrained",tok_itos_pretrained)
            print("tok_itos",tok_itos)
            
            write_log(WORKING_FOLDER,"Remapping tok_itos...")
            tok_itos_new = np.concatenate((tok_itos_pretrained,np.setdiff1d(tok_itos,tok_itos_pretrained)),axis=0)
            tok_stoi_new = {s:i for i,s in enumerate(tok_itos_new)}

            tok_itos_map = np.zeros(len(tok_itos),np.int32)
            for i,t in enumerate(tok_itos):
                tok_itos_map[i]=tok_stoi_new[t]
            np.save(WORKING_FOLDER/'tok_itos.npy',tok_itos_new)
            tok_itos = tok_itos_new
            tok =  np.array([[tok_itos_map[x] for x in t] for t in tok])
            np.save(WORKING_FOLDER/'tok.npy',tok)
    print("tok ito", tok_itos, tok_itos_pretrained)
    #determine pad_idx
    pad_idx = int(np.where(tok_itos=="_pad_")[0][0])

    label_itos = np.load(WORKING_FOLDER/'label_itos.npy',allow_pickle=True)

    label_itos_old = np.load(PRETRAINED_FOLDER/'label_itos.npy')
    if(len(label_itos)!= len(label_itos_old) or (label_itos != label_itos_old).any()):
        print("Warning: label_itos of both models do not coincide")

    trn_toks = tok[train_IDs]
    val_toks = tok[val_IDs]
    trn_labels = label[train_IDs]
    val_labels = label[val_IDs]
    
    assert(len(test_IDs)>0)
    test_toks = tok[test_IDs]
    if(clas):
        test_labels = label[test_IDs]  
        
    print("number of tokens in vocabulary:",len(tok_itos),"\ntrain/val/total sequences:",len(trn_toks),"/",len(val_toks),"/",len(trn_toks)+len(val_toks))

    itos={i:x for i,x in enumerate(tok_itos)}
    vocab=Vocab(itos)
    print(itos)
    ######################
    #set config and arch
    if(kwargs["arch"]== "AWD_LSTM"):
        arch = AWD_LSTM
        config_clas = awd_lstm_clas_config.copy()
        config_clas["emb_sz"]=kwargs["emb_sz"]
        config_clas["n_hid"]=kwargs["nh"]
        config_clas["n_layers"]=kwargs["nl"]
        config_clas["pad_token"]=pad_idx

    #data block api
    src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]), TextList(items=test_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]))
    src = src.label_from_lists(trn_labels,test_labels, classes=label_itos, label_cls=(None), processor=[])
    data_clas_test= src.databunch(bs=kwargs["bs"],pad_idx=pad_idx)

    learn = text_classifier_learner(data_clas_test, arch, config=config_clas, pretrained=False, max_len=kwargs["max_len"], metrics=[])
    for k in kwargs["metrics"]:
        if(k=="accuracy"):
            learn.metrics.append(accuracy)
        elif(k=="macro_f1"):
            macro_f1 = metric_func(partial(fbeta_score, beta=1, average='macro'), "macro_f1", None ,one_hot_encode_target=False, argmax_pred=True)
            learn.metrics.append(macro_f1)


    dl = list(data_clas_test.single_dl)
    random.shuffle(dl)
    target_seqs = dl[:num_seqs]
    return learn, target_seqs, vocab

def load_pretrained_model(learn, kwargs):
    print("Loading model ",kwargs["model_filename_prefix"]+"_3")
    learn.load(kwargs["model_filename_prefix"]+"_3")
    print("Loaded Successfully")

    return learn.model

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

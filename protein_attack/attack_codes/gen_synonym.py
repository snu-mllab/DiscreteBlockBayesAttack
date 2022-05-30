from collections import defaultdict

def get_amino_info():
    name2alph = {}
    alph2name = {}  
    with open('attack_codes/amino', "r") as f:
        lines = f.readlines()
        for line in lines:
            name, alphabet = line[:-1].split(" ")
            name2alph[name] = alphabet
            alph2name[alphabet] = name
    return name2alph, alph2name

def get_dist_mat():
    with open('attack_codes/dist', "r") as f:
        lines = f.readlines()
        first_line = lines[0]
        row_name = first_line[1:-1].split("\t")
        column_name = []
        dist_mat = defaultdict(dict)

        for line in lines[1:]:
            line_split = line[:-1].split("\t")
            column_name.append(line_split[0])
            for i, num in enumerate(line_split[1:-1]):
                col = line_split[0]
                row = row_name[i]
                if row == col:
                    dist_mat[row][col] = 0.0
                elif num == '.':
                    dist_mat[row][col] = float('inf')
                else:
                    dist_mat[row][col] = int(num)/1000
    return dist_mat

def get_synonym(vocab, eps=0.15):
    name2alph, alph2name = get_amino_info()
    dist_mat = get_dist_mat()
    stoi = {k:v for v,k in vocab.itos.items()}

    syndict = defaultdict(list)
    for i in vocab.itos:
        alph = vocab.itos[i]

        if alph in ['_mask_','_bos_','_pad_']:
            syndict[i].append(i)
        else:
            syndict[i].append(i)

            source = alph2name[alph]

            if source in dist_mat:
                for dest in dist_mat[source]:
                    if dist_mat[source][dest] < eps:
                        alph_dest = name2alph[dest]
                        i_dest = stoi[alph_dest]
                        if i_dest!=i: syndict[i].append(i_dest)
            elif source in ['Asx','Glx']:
                if source == 'Asx':
                    source1, source2 = 'Asp','Asn'
                elif source == 'Glx':
                    source1, source2 = 'Glu','Gln'
                for dest in dist_mat['Asp']:
                    if dist_mat[source1][dest] < eps and dist_mat[source2][dest] < eps:
                        alph_dest = name2alph[dest]
                        i_dest = stoi[alph_dest]
                        syndict[i].append(i_dest)
            elif source in ['Xaa','Pyl','Sec']:
                pass
            else:
                raise ValueError
    return syndict

if __name__ == '__main__':
    from fastai.text import Vocab
    import numpy as np
    
    tok_itos = np.load('protein_codes/datasets/clas_ec/clas_ec_ec50_level1/tok_itos.npy',allow_pickle=True)
    itos={i:x for i,x in enumerate(tok_itos)}
    vocab=Vocab(itos)

    syndict= get_synonym(vocab)
    for key, l_ in syndict.items():
        print(key, l_)
   
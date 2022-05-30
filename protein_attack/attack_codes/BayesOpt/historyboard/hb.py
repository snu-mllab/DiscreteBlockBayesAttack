import torch
class HistoryBoard(object):
    def __init__(self, orig_text, transformer, enc_model):
        self.orig_text = orig_text
        setattr(orig_text, 'traj', [])
        self.transformer = transformer
        self.enc_model = enc_model
        self.eval_texts = None
        self.eval_results = None
        self.eval_X = None
        self.eval_Y = None 
        self.numbering_dict = dict()
        self.len_text = len(orig_text.words)

    def best_of_hamming(self, text, distance, mind=0):
        targets = []
        for idx, eval_text in enumerate(self.eval_texts):
            hamming_distance = self._hamming(text, eval_text)
            if hamming_distance <= distance and hamming_distance >= mind:
                targets.append([idx, self.eval_Y[idx][0]])
        if len(targets):
            best_ind, best_score = sorted(targets, key = lambda x: -x[1])[0]
            return self.eval_texts[best_ind], best_score, best_ind
        else:
            return None, -1, None

    def best_in_history(self):
        return self.topk_in_history(1)
            
    def topk_in_history_with_fixed_indices(self, k, cur_indices, fix_indices):
        cur_inds = cur_indices.view(-1)
        hist_inds = self.eval_X_num
        target_inds = (cur_inds[fix_indices] == hist_inds[:,fix_indices]).all(dim=1).nonzero(as_tuple=True)[0]
        
        y = self.eval_Y[target_inds].view(-1)
        _, indices_ = torch.topk(y, min(y.shape[0],k))
        indices = [int(target_inds[ind].item()) for ind in indices_]
        topk_X = self.eval_X[indices]
        topk_Y = self.eval_Y[indices]
        topk_texts = [self.eval_texts[ind] for ind in indices]
        modif = [self._hamming(self.orig_text, txt) for txt in topk_texts]
        return topk_X, topk_Y, modif, indices

    def topk_in_history(self, k):
        _, indices = torch.topk(self.eval_Y.view(-1), min(self.eval_Y.shape[0],k))
        topk_X = self.eval_X[indices]
        topk_Y = self.eval_Y[indices]
        topk_texts = [self.eval_texts[ind] for ind in indices]
        modif = [self._hamming(self.orig_text, txt) for txt in topk_texts]
        return topk_X, topk_Y, modif, indices
    
    def best_in_recent_history(self, num):
        return self.topk_in_recent_history(1, num)

    def topk_in_recent_history(self, k, num):
        _, indices = torch.topk(self.eval_Y[-num:].view(-1), min(self.eval_Y.shape[0],k))
        topk_X = self.eval_X[-num:][indices]
        topk_Y = self.eval_Y[-num:][indices]
        topk_texts = [self.eval_texts[-num:][ind] for ind in indices]
        modif = [self._hamming(self.orig_text, txt) for txt in topk_texts]
        indices = [-num + ind for ind in indices]
        return topk_X, topk_Y, modif, indices

    def _hamming(self, text1, text2):
        return sum([w1!=w2 for w1, w2 in zip(text1.words, text2.words)])

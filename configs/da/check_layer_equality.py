import numpy as np
import pickle
import os
import pdb
from tabulate import tabulate

def load_pickle(path):
    out = pickle.load(open(path,'rb'))
    return out

def check(seed):
    model = 'epoch534'

    main_path = os.path.join('train_log_original_{}'.format(seed),'test_images','Vegas_bn')

    source_path = os.path.join(main_path,'bnAll_before_{}.pth.pickle'.format(model))
    source_ = load_pickle(source_path)
    table=[]
            
    for layer in range(1,61,1):
        upto_k_path = os.path.join(main_path,'bnAll_after_{}_100000000_{}.pickle'.format(model, layer))
        at_k_path = os.path.join(main_path,'bnAll_after_k_{}_100000000_{}.pickle'.format(model, layer))

        upto_k = load_pickle(upto_k_path)
        at_k = load_pickle(at_k_path)
        #table=[]
        
        atkm = np.mean(at_k[(layer-1,'mean')])
        sourcem = np.mean(source_[(layer-1,'mean')])
        table.append([layer, sourcem, atkm, atkm/sourcem ])
        
        if 0: #for l in range(1,60,10):
            matching = np.sqrt(np.sum(at_k[(l,'mean')]- upto_k[(l,'mean')])**2)
            matching_s_upto = np.sqrt(np.sum(source_[(l,'mean')] - upto_k[(l,'mean')])**2)
            matching_s_at = np.sqrt(np.sum(source_[(l,'mean')] - at_k[(l,'mean')])**2)

            table.append([layer, l, (matching_s_upto), (matching_s_at),  (matching) ])
           
            if 0: #l > layer + 5:
                print('\n')
                break

        #print(tabulate(table, headers=("K", "Layer","Source-UptoK","Source-AtK", "UptoK-AtK"), tablefmt="pipe"))
    print(tabulate(table, headers=("K","Source mean","AtK mean", "Relative"), tablefmt="pipe"))
    pdb.set_trace()

        
if __name__ == "__main__":
        seed = 0
        check(seed)

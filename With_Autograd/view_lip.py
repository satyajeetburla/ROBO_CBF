import os
import pickle


fname = os.path.join('./results-tmp', 'lip_a.npy')
with open(fname, 'rb') as handle:
    lip_a = pickle.load(handle)
fname = os.path.join('./results-tmp', 'lip_b.npy')
with open(fname, 'rb') as handle:
    lip_b = pickle.load(handle)
    
print(lip_a,lip_b)

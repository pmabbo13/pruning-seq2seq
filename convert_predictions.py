import os
import pickle
import torch
 
directory = 'predictions'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if f == 'predictions/bart-base-cnn-ca-mag-0.1-predictions':
        continue
    if os.path.isfile(f):
        print(f"Saving {f}")
        all_preds = pickle.load(open(f, "rb"))
        cpu_preds = [pred.to('cpu') for pred in all_preds]
        torch.save(cpu_preds, f)
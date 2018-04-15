import numpy as np
import torch
from torch.utils.data import Dataset

def max_size(data):
    """larger sequence length in the data set"""
    sizes = [len(seq) for seq in data]
    return max(sizes)

def purify(strokes,max_len=200):
    """removes to small or too long sequences + removes large gaps"""
    data = []
    for seq in strokes:
        if len(seq[:,0]) <= max_len and len(seq[:,0])>10:
            seq = np.minimum(seq, 1000)
            seq = np.maximum(seq, -1000)
            seq = np.array(seq, dtype=np.float32)
            data.append(seq)
    return data

def calculate_normalizing_scale_factor(strokes):
    """Calculate the normalizing factor explained in appendix of sketch-rnn."""
    data = []
    for i in range(len(strokes)):
        for j in range(len(strokes[i])):
            data.append(strokes[i][j, 0])
            data.append(strokes[i][j, 1])
    data = np.array(data)
    return np.std(data)

def normalize(strokes):
    """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
    data = []
    scale_factor = calculate_normalizing_scale_factor(strokes)
    for seq in strokes:
        seq[:, 0:2] /= scale_factor
        data.append(seq)
    return data

class Sketch(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index])
    
    def __len__(self):
        return len(self.dataset)

def collate_fn_(batch_data):
    bs = len(batch_data)
    max_len = max([s.size(0) for s in batch_data])
    output = torch.zeros(bs,max_len+1,5)
    output[:,0,2] = 1
    
    for i in range(bs):
        s = batch_data[i].size(0)
        output[i, 1:s+1, :2] = batch_data[i][:,:2]
        output[i, 1:s+1, 2] = 1 - batch_data[i][:, 2]
        output[i, 1:s+1, 3] = batch_data[i][:, 2]
        output[i, s:, 4] = 1
        output[i, s, 2:4] = 0
    return output

def collate_fn_2(batch_data):
    bs = len(batch_data)
    max_len = max([s.size(0) for s in batch_data])
    output = torch.zeros(bs,max_len+1,3)
    
    for i in range(bs):
        end = batch_data[i].size(0)
        output[i, 1:1+end, :3] = batch_data[i][:,:3]
        output[i, end:, 2] = 2 
    return output
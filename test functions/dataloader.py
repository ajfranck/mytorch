import urllib3
import collections
import re
import torch
import math
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time 
from IPython import display
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# download and process dataset
shakespeare = 'http://www.gutenberg.org/files/100/100-0.txt'
http = urllib3.PoolManager()
text = http.request('GET', shakespeare).data.decode('utf-8')
raw_dataset = ' '.join(re.sub('[^A-Za-z]+', ' ', text).lower().split())
idx_to_char = list(set(raw_dataset))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
corpus_indices = [char_to_idx[char] for char in raw_dataset]
sample = corpus_indices[:20]

# print some information about the dataset and vocab
# print('number of characters: ', len(raw_dataset))
# print(raw_dataset[0:70])
# print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
# print('indices:', sample)

# create train and test datasets
train_indices = corpus_indices[:-10000]#0]
test_indices = corpus_indices[-10000:]

input_length=5
vocab = 27

class TextDataset(Dataset):
    def __init__(self, data, input_length, vocabSz, delay=1):
        self.data = data
        self.input_length = input_length
        self.vocabSz = vocabSz
        self.delay = delay

    def __getitem__(self, index): 
        # edit data size in here~! 
        
        input_data = self.data[index : index + self.input_length]
        input_data = F.one_hot(torch.tensor(input_data), num_classes=self.vocabSz)
        # Our label is a single character, delayed by 'delay' from the end of the input sequence
        if(index + self.input_length + self.delay >= len(self.data)-5):
            label = self.data[index + self.input_length + self.delay - 1]
            label = F.one_hot(torch.tensor(label), num_classes=self.vocabSz)
        else:
            label = self.data[index + self.input_length + self.delay]
            label = F.one_hot(torch.tensor(label), num_classes=self.vocabSz)

        return input_data.float(), label.float()

    def __len__(self):
        return len(self.data) - self.input_length

# train_dataset = torch.tensor(np.reshape(train_indices, (-1, input_length+1)))
# test_dataset = torch.tensor(np.reshape(test_indices, (-1, input_length+1)))

train_dataset = TextDataset(train_indices[:1000], input_length, vocab)
test_dataset = TextDataset(test_indices[:1000], input_length, vocab)


# create dataloaders
batch_size = 40

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


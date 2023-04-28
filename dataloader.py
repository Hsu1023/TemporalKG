#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *

class TrainDataset(Dataset):
    def __init__(self, datasetInfo, trainMode, filter_falseNegative=False, negative_sample_size=64):
        self.datasetName          = datasetInfo['datasetName']
        self.len                  = datasetInfo['train_len']
        self.nentity              = datasetInfo['nentity']
        self.nrelation            = datasetInfo['nrelation']
        self.count                = datasetInfo['train_count']        # subsampling_weight
        self.entity_dict          = datasetInfo['entity_dict']        # for ogbl-biokg dataset
        self.triples              = datasetInfo['train_triples']      # list of (h,r,t)
        self.indexing_tail        = datasetInfo['indexing_tail']
        self.trainMode            = trainMode
        self.negative_sample_size = negative_sample_size
        self.filter_falseNegative = filter_falseNegative

        assert self.trainMode in ['negativeSampling', '1VsAll', 'kVsAll']
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        head, relation, tail, time  = self.triples[idx]
        subsampling_weight   = self.count[idx]    
        positive_sample      = torch.LongTensor((head, relation, tail, time))
        filter_mask          = torch.Tensor([-1])        

        # generate negative samples
        if self.trainMode == 'negativeSampling':
            # non-redundant sampling
            negative_sample = torch.randperm(self.nentity)[:self.negative_sample_size]

            if self.filter_falseNegative:
                filter_mask = torch.from_numpy(
                    np.in1d(negative_sample, self.indexing_tail[idx], invert=True)
                    ).int()

        else: 
            # 1VsAll or kVsAll, no needs for generating indexes
            negative_sample = torch.Tensor([-1])

        # generate labels (0/1) for 1(k) Vs All training mode
        if self.trainMode == 'kVsAll':
            tail_peers         = self.indexing_tail[idx]
            labels             = torch.zeros(self.nentity)
            labels[tail_peers] = 1

        else:
            labels = torch.Tensor([-1])

        return positive_sample, negative_sample, labels, filter_mask, subsampling_weight

    @staticmethod
    def collate_fn(data):
        positive_sample    = torch.stack([_[0] for _ in data], dim=0)
        negative_sample    = torch.stack([_[1] for _ in data], dim=0)
        labels             = torch.stack([_[2] for _ in data], dim=0)
        filter_mask        = torch.stack([_[3] for _ in data], dim=0)
        subsampling_weight = torch.cat([_[4]   for _ in data], dim=0)

        return positive_sample, negative_sample, labels, filter_mask, subsampling_weight
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
        
class TestDataset(Dataset):
    def __init__(self, split, args, random_sampling=False, entity_dict=None):
        self.datasetName     = args.datasetInfo['datasetName']
        self.neg_size        = args.neg_size_eval_train if random_sampling else -1
        self.random_sampling = random_sampling
        
        if split == 'validate':
            self.triples         = args.datasetInfo['valid_triples']
        elif split == 'test':
            self.triples         = args.datasetInfo['test_triples']
        else:
            self.triples         = args.datasetInfo['train_triples']
        
        if self.datasetName not in ['ogbl-biokg', 'ogbl-wikikg2']:
            if split == 'validate':
                self.filteredSamples = args.datasetInfo['valid_negSamples']
            elif split == 'test':
                self.filteredSamples = args.datasetInfo['test_negSamples']
            else:
                self.filteredSamples = args.datasetInfo['train_negSamples']

        self.len             = len(self.triples['head']) if (self.datasetName in ['ogbl-biokg', 'ogbl-wikikg2']) else len(self.triples)
        self.nentity         = args.datasetInfo['nentity']
        self.nrelation       = args.datasetInfo['nrelation']
        self.entity_dict     = args.datasetInfo['entity_dict']

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        
        head, relation, tail, time = self.triples[idx]
        filter_bias          = self.filteredSamples[idx]
        positive_sample      = torch.LongTensor((head, relation, tail, time))     

        return positive_sample, filter_bias

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, negative_sample
    
    @staticmethod
    def collate_fn_with_bias(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        filter_bias     = torch.stack([_[1] for _ in data], dim=0)
        return positive_sample, filter_bias

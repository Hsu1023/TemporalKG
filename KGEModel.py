import time
import os
import random
import itertools
import logging
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import DataLoader
from   collections import defaultdict
from   tqdm import tqdm
from   dataloader import TestDataset
from   utils import *


class KGEModel(nn.Module):
    def __init__(self, model_name, dataset_name, nentity, nrelation, ntime, id2timestr, params_dict, config, evaluator=None):
        super(KGEModel, self).__init__()
        '''
            KGEModel class
            components:
                - definition of KGE models 
                - train and test functions
        '''
        # checking parameters
        if model_name not in ['TTransE', 'ComplEx', 'TComplEx', 'TNTComplEx', 'TATransE', 'TADistMult', 'DETransE', 'DEDistMult', 'DESimplE']:
            raise ValueError('model %s not supported' % model_name)

        # build model
        self.model_name           = model_name
        self.dataset              = dataset_name.lower()
        self.config               = config

        self.nentity              = nentity
        self.nrelation            = nrelation if config.training_strategy.shareInverseRelation else 2*nrelation
        self.ntime                = ntime
        self.id2timestr           = id2timestr
        self.hidden_dim           = params_dict['dim']
        self.epsilon              = 2.0
        self.gamma                = nn.Parameter(torch.Tensor([params_dict['gamma']]), requires_grad=False)
        self.embedding_range      = nn.Parameter(torch.Tensor([params_dict['embedding_range']]), requires_grad=False)

        # set relation dimension according to specific model
        if model_name == 'RotatE':
            self.relation_dim = int(self.hidden_dim / 2)
        elif model_name == 'RESCAL':
            self.relation_dim = int(self.hidden_dim ** 2)
        else:
            self.relation_dim = self.hidden_dim


        if self.model_name in ['DESimplE', 'DETransE', 'DEDistMult']:
            self.t_dim = int(self.hidden_dim * 0.1) # dynamic dim
            self.s_dim = self.hidden_dim - self.t_dim # static dim
            
            if self.model_name in ['DESimplE']:
                self.yy_a = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.yy_w = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.yy_b = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.mm_a = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.mm_w = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.mm_b = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.dd_a = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.dd_w = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))
                self.dd_b = nn.Parameter(torch.zeros(self.nentity * 2, self.t_dim))  
                self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation * 2, self.relation_dim))  
                self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity * 2, self.s_dim))
            else:
                self.yy_a = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.yy_w = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.yy_b = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.mm_a = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.mm_w = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.mm_b = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.dd_a = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.dd_w = nn.Parameter(torch.zeros(self.nentity, self.t_dim))
                self.dd_b = nn.Parameter(torch.zeros(self.nentity, self.t_dim))  
                self.relation_embedding = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
                self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity, self.s_dim))
            self.time_act = torch.sin
        else:
            self.entity_embedding     = nn.Parameter(torch.zeros(self.nentity, self.hidden_dim))
            self.relation_embedding   = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
            if self.model_name in ['TTransE']:
                self.time_embedding       = nn.Parameter(torch.zeros(self.ntime, self.hidden_dim))
            elif self.model_name in ['TComplEx', 'TNTComplEx']:
                self.time_embedding       = nn.Parameter(torch.zeros(self.ntime, self.hidden_dim))
                self.aux_embedding        = nn.Parameter(torch.zeros(self.nrelation, self.relation_dim))
            elif self.model_name in ['TATransE', 'TADistMult']:
                self.ntem = 32
                self.tem_embedding        = nn.Parameter(torch.zeros(self.ntem, self.hidden_dim))  
                self.lstm = nn.LSTM(input_size=self.relation_dim, hidden_size=self.relation_dim, num_layers=1, batch_first=True)

        self.evaluator            = evaluator
        
        # read essential training config (from global_config)
        self.dropoutRate          = config.training_strategy.dropoutRate
        self.dropout              = nn.Dropout(p=self.dropoutRate)
        self.training_mode        = config.training_strategy.training_mode
        self.shareInverseRelation = config.training_strategy.shareInverseRelation
        self.label_smooth         = config.training_strategy.label_smooth
        self.loss_name            = config.training_strategy.loss_function
        self.uni_weight           = config.training_strategy.uni_weight
        self.adv_sampling         = config.training_strategy.negative_adversarial_sampling
        self.filter_falseNegative = config.training_strategy.filter_falseNegative
        self.adv_temperature      = params_dict['advs']
        self.regularizer          = params_dict['regularizer']   # FRO NUC DURA None
        self.regu_weight          = params_dict['regu_weight']

        # setup candidate loss functions
        self.KLLoss               = nn.KLDivLoss(size_average=False)
        self.MRLoss               = nn.MarginRankingLoss(margin=float(self.gamma), reduction='none')
        self.CELoss               = nn.CrossEntropyLoss(reduction='none')
        self.BCELoss              = nn.BCEWithLogitsLoss(reduction='none')
        self.weightedBCELoss      = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.Tensor([self.nentity]))

        # initialize embedding
        self.init_embedding(config.training_strategy.initializer)
        self.model_func = {
            'TransE':   self.TransE,
            'DistMult': self.DistMult,
            'ComplEx':  self.ComplEx,
            'TComplEx': self.TComplEx,
            'TTransE': self.TTransE,
            'TNTComplEx': self.TNTComplEx,
            'TATransE': self.TAModel,
            'TADistMult': self.TAModel,
            'DESimplE': self.DEModel,
            'DETransE': self.DEModel,
            'DEDistMult': self.DEModel,
        }
        
    def init_embedding(self, init_method):
        embedding_to_init = [self.entity_embedding, self.relation_embedding]
        if self.model_name in ['TATransE', 'TADistMult']:
            embedding_to_init += [self.tem_embedding]
        elif self.model_name in ['DESimplE', 'DETransE', 'DEDistMult']:
            embedding_to_init += [self.yy_a, self.yy_w, self.yy_b, self.mm_a, self.mm_w, self.mm_b, self.dd_a, self.dd_w, self.dd_b]
        elif self.model_name in ['TTransE', 'TComplEx']:
            embedding_to_init += [self.time_embedding]
        elif self.model_name in ['TNTComplEx', 'DESimplE', 'SimplE']:
            embedding_to_init += [self.time_embedding, self.aux_embedding]


        if init_method == 'uniform':
            # Fills the input Tensor with values drawn from the uniform distribution
            for tensor in embedding_to_init:
                nn.init.uniform_(tensor, a=-self.embedding_range.item(), b=self.embedding_range.item())
        
        elif init_method == 'xavier_normal':
            for tensor in embedding_to_init:
                nn.init.xavier_normal_(tensor=tensor)

        elif init_method == 'normal':
            # Fills the input Tensor with values drawn from the normal distribution
            for tensor in embedding_to_init:
                nn.init.normal_(tensor=tensor, mean=0.0, std=self.embedding_range.item())

        elif init_method == 'xavier_uniform':
            for tensor in embedding_to_init:
                nn.init.xavier_uniform_(tensor=tensor)

        return

    def forward(self, sample, mode='single'):
        '''
            3 available modes: 
                - single     : for calculating positive scores
                - neg_sample : for negative sampling
                - all        : for 1(k) vs All training 
        '''
        head_index, relation_index, tail_index, time_index = sample
        inv_relation_mask = None
        if self.model_name != 'DESimplE':
            inv_relation_mask = torch.where(relation_index >= self.nrelation) if self.shareInverseRelation else None
            relation_index    = relation_index % self.nrelation if self.shareInverseRelation else relation_index          

        if self.model_name in ['TADistMult', 'TATransE']: 
            head              = self.dropout(self.entity_embedding[head_index])
            relation          = self.dropout(self.relation_embedding[relation_index])
            tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])
            time_tokens = self.TA_get_time_tokens(time_index) # (B, 4+2+2)
            time_input = self.dropout(self.tem_embedding[time_tokens]) # (B, 8, dim)
            score = self.TAModel(head, relation, tail, time_input, inv_relation_mask=inv_relation_mask, mode=mode)

        elif self.model_name in ['DETransE', 'DEDistMult', 'DESimplE']:
            if self.model_name == 'DESimplE':
                score = self.DESimplE(head_index, relation_index, tail_index, time_index, inv_relation_mask, mode=mode)
            else:
                score = self.DEModel(head_index, relation_index, tail_index, time_index, inv_relation_mask, mode=mode)
        elif self.model_name in ['TTransE', 'TComplEx']:
            head              = self.dropout(self.entity_embedding[head_index])
            relation          = self.dropout(self.relation_embedding[relation_index])
            tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])
            time = self.dropout(self.time_embedding[time_index])
            score = self.model_func[self.model_name](head, relation, tail, time,inv_relation_mask=inv_relation_mask, mode=mode)

        elif self.model_name in ['TNTComplEx']:
            head              = self.dropout(self.entity_embedding[head_index])
            relation          = self.dropout(self.relation_embedding[relation_index])
            tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])
            time = self.dropout(self.time_embedding[time_index])
            aux  = self.dropout(self.aux_embedding[relation_index])
            score = self.model_func[self.model_name](head, relation, tail, time, aux, inv_relation_mask=inv_relation_mask, mode=mode)
        else:
            head              = self.dropout(self.entity_embedding[head_index])
            relation          = self.dropout(self.relation_embedding[relation_index])
            tail              = self.dropout(self.entity_embedding if mode == 'all' else self.entity_embedding[tail_index])
            score = self.model_func[self.model_name](head, relation, tail, inv_relation_mask=inv_relation_mask, mode=mode)
        
        return score
    
    def DE_get_time(self, time_index_):
        time_index = time_index_.cpu()
        year = torch.zeros((time_index.shape[0]), dtype=torch.long, requires_grad=False)
        month = torch.zeros((time_index.shape[0]), dtype=torch.long, requires_grad=False)
        day = torch.zeros((time_index.shape[0]), dtype=torch.long, requires_grad=False)
        for idx, t in enumerate(time_index): 
            yy, mm, dd = str(self.id2timestr[t.item()]).split('-') 
            year[idx] = int(yy)
            month[idx] = int(mm)
            day[idx] = int(dd)
        return year.reshape(-1).cuda(), month.reshape(-1).cuda(), day.reshape(-1).cuda()
    
    def DE_get_time_embeddings(self, index, year, month, day):
        y = self.dropout(self.yy_a[index]) * self.time_act(self.dropout(self.yy_w[index]) * year + self.dropout(self.yy_b[index]))
        m = self.dropout(self.mm_a[index]) * self.time_act(self.dropout(self.mm_w[index]) * month + self.dropout(self.mm_b[index]))
        d = self.dropout(self.dd_a[index]) * self.time_act(self.dropout(self.dd_w[index]) * day + self.dropout(self.dd_b[index]))
        return y + m + d

    def DEModel(self, head_index, relation_index, tail_index, time_index, inv_relation_mask, mode='single'):
        year, month, day = self.DE_get_time(time_index)
        year, month, day = year.reshape(-1,1), month.reshape(-1,1), day.reshape(-1,1)
        head_t = self.DE_get_time_embeddings(head_index, year, month, day) # [B, 0.1 dim]
        head_s = self.dropout(self.entity_embedding[head_index]) # [B, 0.9 dim]
        head = torch.cat((head_t, head_s), dim=1) # [B, dim]

        year, month, day = year.reshape(-1, 1, 1), month.reshape(-1, 1, 1), day.reshape(-1, 1, 1)
        if mode == 'all':
            tail_index = torch.arange(self.nentity).cuda()
        tail_t = self.DE_get_time_embeddings(tail_index, year, month, day) # [B, pos/neg, 0.1 dim]
        tail_s = self.dropout(self.entity_embedding[tail_index])# [B, pos/neg, 0.9d]; all : [nentity, 0.9d]
        if mode == 'all':
            tail_s = tail_s.unsqueeze(0).repeat(tail_t.shape[0], 1, 1) # [B, nentity, 0.9 dim]
        tail = torch.cat((tail_t, tail_s), dim=2) # [B, pos/neg, dim]

        relation = self.dropout(self.relation_embedding[relation_index])

        # DE-XX -> XX
        return self.model_func[self.model_name[2:]](head, relation, tail, inv_relation_mask=inv_relation_mask, mode='negativeSampling' if mode!='single' else mode)
    
    def DESimplE(self, head_index, relation_index, tail_index, time_index, inv_relation_mask, mode='single'):
        assert self.shareInverseRelation
        year, month, day = self.DE_get_time(time_index)
        year, month, day = year.reshape(-1,1), month.reshape(-1,1), day.reshape(-1,1)
        # head_index for head
        head1_t = self.DE_get_time_embeddings(head_index, year, month, day) # [B, 0.1 dim]
        head1_s = self.dropout(self.entity_embedding[head_index]) # [B, 0.9 dim]
        head1 = torch.cat((head1_t, head1_s), dim=1) # [B, dim]
        # head_index for tail
        head2_t = self.DE_get_time_embeddings(head_index + self.nentity, year, month, day) # [B, 0.1 dim]
        head2_s = self.dropout(self.entity_embedding[head_index + self.nentity]) # [B, 0.9 dim]
        head2 = torch.cat((head2_t, head2_s), dim=1) # [B, dim]       

        year, month, day = year.reshape(-1, 1, 1), month.reshape(-1, 1, 1), day.reshape(-1, 1, 1)
        # tail_index for head
        if mode == 'all':
            tail_index = torch.arange(self.nentity).cuda()
        tail1_t = self.DE_get_time_embeddings(tail_index, year, month, day) # [B, pos/neg, 0.1 dim]
        tail1_s = self.dropout(self.entity_embedding[tail_index])# [B, pos/neg, 0.9d]; all : [nentity, 0.9d]
        if mode == 'all':
            tail1_s = tail1_s.unsqueeze(0).repeat(tail1_t.shape[0], 1, 1) # [B, nentity, 0.9 dim]
        tail1 = torch.cat((tail1_t, tail1_s), dim=2) # [B, pos/neg, dim]
        # tail_index for tail
        tail2_t = self.DE_get_time_embeddings(tail_index + self.nentity, year, month, day) # [B, pos/neg, 0.1 dim]
        tail2_s = self.dropout(self.entity_embedding[tail_index + self.nentity])# [B, pos/neg, 0.9d]; all : [nentity, 0.9d]
        if mode == 'all':
            tail2_s = tail2_s.unsqueeze(0).repeat(tail2_t.shape[0], 1, 1) # [B, nentity, 0.9 dim]
        tail2 = torch.cat((tail2_t, tail2_s), dim=2) # [B, pos/neg, dim]

        relation = self.dropout(self.relation_embedding[relation_index])
        relation_inv_index = (relation_index + self.nrelation) % (2 * self.nrelation)
        relation_inv = self.dropout(self.relation_embedding[relation_inv_index])

        if mode == 'all': # head: [B, dim]; tail: [nentity, dim]
            score = ((head1.unsqueeze(1) * relation.unsqueeze(1)) * tail2 +  
                    (head2.unsqueeze(1) * relation_inv.unsqueeze(1)) * tail1) / 2.0 # [B, nentity, dim]
            
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head1, relation, tail1, head2, relation_inv, tail2], [[head1 * relation, tail1 * relation_inv], [head2, tail2]]) ## TODO:regu
            score = ((head1 * relation).unsqueeze(1) * tail2 + (head2 * relation_inv).unsqueeze(1) * tail1) / 2.0
        return score.sum(dim=2) # [B, pos/neg]
    
    def TAModel(self, head, relation, tail, time_input, inv_relation_mask, mode='single'):
        time_input = torch.concat((relation.unsqueeze(1), time_input), dim=1) # (B, 9, dim)
        h0 = torch.zeros((1, head.shape[0], self.hidden_dim), device='cuda:0')
        c0 = torch.zeros((1, head.shape[0], self.hidden_dim), device='cuda:0')
        output, (hn, cn) = self.lstm(time_input, (h0, c0))
        output = output[:, -1, :] # (B, dim)

        # TA-XX -> XX
        return self.model_func[self.model_name[2:]](head, relation, tail, output, inv_relation_mask=[], mode=mode) ## TODO: inv
    
    def TA_get_time_tokens(self, time_index_):
            time_index = time_index_.cpu()
            index_matrix = torch.zeros((time_index.shape[0], 4+2+2), dtype=torch.long, requires_grad=False)
            for idx, t in enumerate(time_index):
                yy, mm, dd = str(self.id2timestr[t.item()]).split('-')
                index_matrix[idx, :4] = torch.tensor([int(y) for y in yy])
                index_matrix[idx, 4:6] = torch.tensor([int(m) + 10 for m in mm])
                index_matrix[idx, 6:8] = torch.tensor([int(d) + 22 for d in dd])
            return index_matrix.cuda()
    
    def TransE(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
            (h,r,t):     h + r = t
            (t,INV_r,h): t + (-r) = h, INV_r = -r
            ori: score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        '''
        if self.shareInverseRelation and inv_relation_mask:
            relation[inv_relation_mask] = -relation[inv_relation_mask]

        if mode == 'all': # head: [B, dim]; tail: [nentity, dim]
            score = (head + relation).unsqueeze(1) - tail.unsqueeze(0)
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], [[head + relation], [tail]])
            score = (head + relation).unsqueeze(1) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score
    
    def TTransE(self, head, relation, tail, time, inv_relation_mask, mode='single'):
        if self.shareInverseRelation and inv_relation_mask:
            relation[inv_relation_mask] = -relation[inv_relation_mask]

        if mode == 'all':
            score = (head + relation + time).unsqueeze(1) - tail.unsqueeze(0)
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail, time], [[head + relation + time],[tail]])
            score = (head + relation + time).unsqueeze(1) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        return score
    
    def DistMult(self, head, relation, tail, inv_relation_mask, mode='single'):
        if mode == 'all':
            # for 1(k) vs all: [B, dim] * [dim, N] -> [B, N]
            score = torch.mm(head * relation, tail.transpose(0,1))
        else:
            if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], [[head * relation], [tail]])
            score = torch.sum((head * relation).unsqueeze(1) * tail, dim=-1)

        return score

    def regularizeOnPositiveSamples(self, embeddings, queries):
        '''
        available regularizer: 
            FRO / NUC / DURA / None
        inputs:
            embeddings: heads, relations, tails 
            queries:    combination of heads and relations
        '''

        self.regu = 0      

        if self.regularizer == 'FRO':
            # squared L2 norm
            for embedding in embeddings:
                self.regu += embedding.norm(p = 2)**2 / embedding.shape[0]

        elif self.regularizer == 'NUC':
            # nuclear 3-norm
            for embedding in embeddings:
                self.regu += embedding.norm(p = 3)**3 / embedding.shape[0]

        elif self.regularizer == 'DURA':
            # duality-induced regularizer for tensor decomposition models
            # regu = L2(φ(h,r)) + L2(t)
            phi, b = queries
            for phi_ in phi:
                self.regu += phi_.norm(p = 2)**2 / phi_.shape[0]
            for b_ in b:
                self.regu += b_.norm(p = 2)**2 / b_.shape[0]

        else: 
            # None
            pass

        self.regu *= self.regu_weight
        return

    def ComplEx(self, head, relation, tail, inv_relation_mask, mode='single'):
        '''
        INV_r = Conj(r)
        '''
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)

        if self.shareInverseRelation:
            im_relation = im_relation.clone()
            im_relation[inv_relation_mask] = - im_relation[inv_relation_mask]

        re_hrvec = re_head * re_relation - im_head * im_relation
        im_hrvec = re_head * im_relation + im_head * re_relation
        hr_vec   = torch.cat([re_hrvec, im_hrvec], dim=-1)

        # regularization on positive samples
        if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail], [[hr_vec], [tail]])

        # <φ(h,r), t> -> score
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score

    def TComplEx(self, head, relation, tail, time, inv_relation_mask, mode='single'):
        '''
        INV_r = Conj(r)
        '''
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        re_time, im_time = torch.chunk(time, 2, dim=-1)
        # re_time, im_time = torch.chunk(torch.ones(time.shape).to('cuda:0'), 2, dim=-1)

        if self.shareInverseRelation:
            im_relation = im_relation.clone()
            im_relation[inv_relation_mask] = - im_relation[inv_relation_mask]

        re_rtvec = re_relation * re_time - im_relation * im_time
        im_rtvec = im_relation * re_time + re_relation * im_time

        re_hrtvec = re_head * re_rtvec - im_head * im_rtvec
        im_hrtvec = im_head * re_rtvec + re_head * im_rtvec
        hr_vec   = torch.cat([re_hrtvec, im_hrtvec], dim=-1)

        # regularization on positive samples
        if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail, time], [[hr_vec],[tail]])

        # <φ(h,r), t> -> score
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score
    
    # regu tricks
    def TNTComplEx(self, head, relation, tail, time, aux, inv_relation_mask, mode='single'):
        '''
        INV_r = Conj(r)
        '''
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_relation, im_relation = torch.chunk(relation, 2, dim=-1)
        re_time, im_time = torch.chunk(time, 2, dim=-1)
        re_rela_static, im_rela_static = torch.chunk(aux, 2, dim=-1)

        if self.shareInverseRelation:
            im_relation = im_relation.clone()
            im_relation[inv_relation_mask] = - im_relation[inv_relation_mask]

        re_rtvec = re_relation * re_time - im_relation * im_time + re_rela_static
        im_rtvec = im_relation * re_time + re_relation * im_time + im_rela_static

        re_hrtvec = re_head * re_rtvec - im_head * im_rtvec
        im_hrtvec = im_head * re_rtvec + re_head * im_rtvec
        hr_vec   = torch.cat([re_hrtvec, im_hrtvec], dim=-1)

        # regularization on positive samples
        if mode == 'single': self.regularizeOnPositiveSamples([head, relation, tail, time, aux], [[hr_vec], [tail]])

        # <φ(h,r), t> -> score
        if mode == 'all':
            score = torch.mm(hr_vec, tail.transpose(0, 1))
        else:
            score = torch.sum(hr_vec.unsqueeze(1) * tail, dim=-1)

        return score

    def saveEmbeddingToFile(self, savePath):
        saveData = {}
        saveData['entity_embedding']   = self.entity_embedding.cpu()
        saveData['relation_embedding'] = self.relation_embedding.cpu()
        logging.info(f'save embedding tensor to: {savePath}')
        pkl.dump(saveData, open(savePath, "wb" ))
        return

    def loadEmbeddingFromFile(self, savePath):
        if not os.path.exists(savePath):
            logging.info(f'[Error] embedding file does not exist: {savePath}')
            return
        data = savePickleReader(savePath)
        self.entity_embedding   = nn.Parameter(data['entity_embedding'])
        self.relation_embedding = nn.Parameter(data['relation_embedding'])
        logging.info(f'successfully loaded pretrained embedding from: {savePath}')
        return

    def train_step(self, model, optimizer, train_iterator):
        # prepare
        model.train()
        optimizer.zero_grad()
        onestep_summary = {}

        # data preparing
        positive_sample, negative_sample, labels, filter_mask, subsampling_weight = next(train_iterator)

        if self.training_mode == '1VsAll':
            labels = torch.zeros(positive_sample.shape[0], self.nentity) 
            labels[list(range(positive_sample.shape[0])), positive_sample[:, 2]] = 1

        # move to device
        positive_sample    = positive_sample.cuda()
        negative_sample    = negative_sample.cuda()
        labels             = labels.cuda()
        filter_mask        = filter_mask.cuda() if self.filter_falseNegative else None
        subsampling_weight = subsampling_weight.cuda()

        # forward
        positive_score = model((positive_sample[:,0], positive_sample[:,1], positive_sample[:,2].unsqueeze(1), positive_sample[:,3]), mode='single')     # [B, 1]
        if self.training_mode == 'negativeSampling':    
            negative_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample, positive_sample[:,3]), mode='neg_sample')               # [B, N_neg]
        else:
            all_score = model((positive_sample[:,0], positive_sample[:,1], negative_sample, positive_sample[:,3]), mode='all')                           # [B, N_neg]
            

        # Margin Ranking Loss (MR)
        if self.loss_name == 'MR':
            # only supporting training mode of negativeSampling
            target = torch.ones(positive_score.size()).cuda()
            loss   = self.MRLoss(positive_score, negative_score, target)
            loss   = (loss * filter_mask).mean(-1) if self.filter_falseNegative else loss.mean(-1)                                  # [B]
            loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())             # [1]

        # Binary Cross Entropy Loss (BCE) 
        elif self.loss_name == 'BCE_mean':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).cuda()
                neg_label = torch.zeros(negative_score.size()).cuda()
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).mean(-1) if self.filter_falseNegative else neg_loss.mean(-1)                   # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum()) 

            else:
                # label smoothing
                labels   = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss     = self.weightedBCELoss(all_score, labels).mean(dim=1)                                                     # [B]
                loss     = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())

        elif self.loss_name == 'BCE_sum':
            if self.training_mode == 'negativeSampling':
                pos_label = torch.ones(positive_score.size()).cuda()
                neg_label = torch.zeros(negative_score.size()).cuda()
                
                # label smoothing
                pos_label = (1.0 - self.label_smooth)*pos_label + (1.0/self.nentity) if self.label_smooth > 0 else pos_label
                neg_label = (1.0 - self.label_smooth)*neg_label + (1.0/self.nentity) if self.label_smooth > 0 else neg_label
                pos_loss  = self.BCELoss(positive_score, pos_label).squeeze(-1)                                                     # [B]
                neg_loss  = self.BCELoss(negative_score, neg_label)                                                                 # [B, N_neg]
                neg_loss  = (neg_loss * filter_mask).sum(-1) if self.filter_falseNegative else neg_loss.sum(-1)                     # [B]
                loss      = pos_loss + neg_loss 
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())   
            else: # 1vsAll or kvsAll
                # label smoothing
                labels = (1.0 - self.label_smooth)*labels + (1.0/self.nentity) if self.label_smooth > 0 else labels
                loss   = self.BCELoss(all_score, labels).sum(dim=1)                                                              # [B]
                loss   = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())

        elif self.loss_name == 'BCE_adv':
            # assert self.training_mode == 'negativeSampling'
            pos_loss = self.BCELoss(positive_score, torch.ones(positive_score.size()).cuda()).squeeze(-1)                          # [B]
            neg_loss = self.BCELoss(negative_score, torch.zeros(negative_score.size()).cuda())                                     # [B, N_neg]
            neg_loss = ( F.softmax(negative_score * self.adv_temperature, dim=1).detach() * neg_loss )

            if self.training_mode == 'negativeSampling' and self.filter_falseNegative:
                neg_loss  = (neg_loss * filter_mask).sum(-1) 
            else:
                neg_loss  =  neg_loss.sum(-1)                                                                                      # [B]

            loss     = pos_loss + neg_loss 
            loss     = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())                                                                             
        
        # Cross Entropy (CE)
        elif self.loss_name == 'CE':
            if self.training_mode == 'negativeSampling':
                # note that filter false negative samples is not supported here
                cat_score = torch.cat([positive_score, negative_score], dim=1)
                labels    = torch.zeros((positive_score.size(0))).long().cuda()
                loss      = self.CELoss(cat_score, labels)
                loss      = loss.mean() if self.uni_weight else ((subsampling_weight * loss).sum() / subsampling_weight.sum())
            
            elif self.training_mode in ['1VsAll', 'kVsAll']:
                loss = self.KLLoss(F.log_softmax(all_score, dim=1), F.normalize(labels, p=1, dim=1))   
                # loss = torch.nn.CrossEntropyLoss(reduce='mean')(all_score, positive_sample[:, 2])
                
        
        if torch.isnan(loss):
            onestep_summary['NAN loss'] = True
            return onestep_summary

        if torch.is_tensor(self.regu) and not (torch.isinf(self.regu) or torch.isnan(self.regu)):
            loss += self.regu

        loss.backward()
        optimizer.step()

        return onestep_summary

    def test_step(self, model, args, split, random_sampling=False):

        '''
        Evaluate the model on test or valid datasets
        '''
        model.eval()
        select_collate_fn = TestDataset.collate_fn if ('ogb' in args.dataset) else TestDataset.collate_fn_with_bias

        # Prepare dataloader for evaluation
        test_dataset = DataLoader(
                TestDataset(split, args, random_sampling), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num), 
                collate_fn=select_collate_fn)

        
        # other datasets
        all_ranking = []
        with torch.no_grad():
            for positive_sample, filter_bias in test_dataset:
                positive_sample = positive_sample.cuda()
                filter_bias     = filter_bias.cuda()

                # forward
                score = model((positive_sample[:,0], positive_sample[:,1], None, positive_sample[:,3]), 'all')
                score = score - torch.min(score, dim=1)[0].unsqueeze(1)
                score *= filter_bias
                
                # explicitly sort all the entities to ensure that there is no test exposure bias
                argsort = torch.argsort(score, dim=1, descending=True)
                positive_arg = positive_sample[:, 2] # indexes of target entities
                
                # obtain rankings for the batch
                tmp_ranking = torch.nonzero(argsort == positive_arg.unsqueeze(1))[:, 1].cpu().numpy() + 1
                all_ranking += list(tmp_ranking)

        # calculate metrics
        all_ranking        = np.array(all_ranking)
        metrics            = {}
        metrics['mrr']     = np.mean(1/all_ranking)
        metrics['mr']      = np.mean(all_ranking)
        metrics['hits@1']  = np.mean(all_ranking<=1)
        metrics['hits@3']  = np.mean(all_ranking<=3)
        metrics['hits@10'] = np.mean(all_ranking<=10)

        return metrics


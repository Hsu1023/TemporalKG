import os
import pickle as pkl                 
import argparse
import os
import sys
import inspect
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix
from utils import *
import heapq

from grakel.utils import graph_from_networkx

from grakel.kernels import ShortestPath,WeisfeilerLehman,VertexHistogram,MultiscaleLaplacian,HadamardCode,SvmTheta,LovaszTheta,RandomWalk,GraphletSampling,Propagation,NeighborhoodHash,NeighborhoodSubgraphPairwiseDistance,OddSth,Propagation,PyramidMatch,RandomWalk,EdgeHistogram

from littleballoffur.edge_sampling import *
from littleballoffur.exploration_sampling import *
from littleballoffur.node_sampling import *

'''
This script is use to generate sampled KG via random walk
'''

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('-data_path', type=str, default='./dataset/')
    parser.add_argument('-sample_ratio', type=float, default=0.2)
    parser.add_argument('-repeat', type=int, default=0)
    parser.add_argument('-perf_file', type=str, default='./results/sampling.txt')
    parser.add_argument('-folder_name', type=str, default=None)
    parser.add_argument('-genpeel', default=False, action='store_true')
    parser.add_argument('-p', type=float, default=1)
    parser.add_argument('-littleball', default=False, action='store_true')
    parser.add_argument('-random_walk', default=False, action='store_true')
    parser.add_argument('-num_starts', type=int, default=10)

    
    return parser.parse_args(args)

def set_logger(log_file):
    '''
    save logs to checkpoint and console
    DEBUG INFO WARNING ERROR CRITICAL
    '''
    
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)   
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    for h in logger.handlers:
        h.setFormatter(formatter)

    return logger

class KGGenerator:
    def __init__(self, mode, payload, nentity, all_triples, sample_ratio, folderName, savePath, entity_dict=None, repeat_num=0, split_ratio=[0.9, 0.05, 0.05]):        
        self.split_ratio  = split_ratio
        self.sample_ratio = sample_ratio
        self.repeat_num   = repeat_num
        self.mode = mode
        self.payload   = payload
        self.nentity = nentity

        # setup graph
        print('==> building nx graph...')
        homoGraph = self.triplesToNxGraph(all_triples)
        diGraph   = self.triplesToNxDiGraph(all_triples)
        print('==> Done!')

        # sampling via random walk
        print('==> start {} sampling...'.format(mode))
        num_nodes = homoGraph.number_of_nodes()
        print('num_nodes',num_nodes,type(num_nodes))
        print('self.sample_ratio',self.sample_ratio,type(self.sample_ratio))
        # print(num_nodes * self.sample_ratio)

        target_num_nodes = int(num_nodes * self.sample_ratio)
        if mode == 'random_walk':
            if payload == 1:
                sampled_nodes = self.random_walk_induced_graph_sampling(homoGraph, target_num_nodes)
            else:
                sampled_nodes = self.multi_starts_random_walk_induced_graph_sampling(homoGraph, target_num_nodes, payload)
             # 得到子图
            sampled_graph = diGraph.subgraph(sampled_nodes)
        elif mode == 'genpeel':
            sampled_nodes = self.Genpeel_graph_sampling(homoGraph, target_num_nodes, payload)
             # 得到子图
            sampled_graph = diGraph.subgraph(sampled_nodes)
        elif mode == 'littleball':
            loc = locals()
            exec('sampler = {}(number_of_nodes=target_num_nodes)'.format(folderName))
            sampler = loc['sampler']

            # add node
            lack_nodes = set(range(self.nentity)) - set(homoGraph.nodes)
            homoGraph.add_nodes_from(list(lack_nodes))

            sampled_graph = sampler.sample(homoGraph.to_undirected())
            sampled_graph = nx.Graph(sampled_graph) # unfrozen the graph
            sampled_graph.remove_nodes_from(list(lack_nodes))
            sampled_graph = diGraph.subgraph(sampled_graph.nodes)
        else:
            raise('no such sampling method')

        # for kernel in ['PyramidMatch','VertexHistogram','EdgeHistogram','NeighborhoodHash']:
            
        #     G = graph_from_networkx([diGraph, sampled_graph],node_labels_tag='entity',edge_labels_tag='relation')
        #     loc = locals()
        #     exec("graphKernel = {}(normalize=True)".format(kernel))
        #     graphKernel = loc['graphKernel']
        #     result = graphKernel.fit_transform(G)
        #     logging.info('==> kernel: {}, correlation: {}'.format(kernel, float(result[0][1])))

        logging.info('==> Done!')

        # build sampled KG
        self.all_triples = []
        self.relations   = []
        self.entities    = []
        for edge in list(sampled_graph.edges(data=True)):
            h,t = edge[0], edge[1]
            r = edge[2]['relation']
            self.all_triples.append((h,r,t))
            self.relations.append(r)
            self.entities.append(h)
            self.entities.append(t)

        # assign new index to entities/relation
        self.entities  = sorted(list(set(self.entities)))
        self.nentity   = len(self.entities)
        self.relations = sorted(list(set(self.relations)))
        self.nrelation = len(self.relations)
        self.ntriples  = len(self.all_triples)
        self.sparsity  = self.ntriples / (self.nentity * self.nentity * self.nrelation)
        self.entity_mapping_dict = {}
        self.relation_mapping_dict = {}

        logging.info('dataset={}, nentity={}, sampled ratio={}, sparsity={}'.format(
            args.dataset, self.nentity, self.sample_ratio, self.sparsity))
        print('dataset={}, nentity={}, sampled ratio={}, sparsity={}'.format(
            args.dataset, self.nentity, self.sample_ratio, self.sparsity))
       
        # key:   origin index
        # value: new assigned index
        for idx in range(self.nentity):
            self.entity_mapping_dict[self.entities[idx]] = idx
        for idx in range(self.nrelation):
            self.relation_mapping_dict[self.relations[idx]] = idx

        # get new triples via entitie_mapping_dict
        self.all_new_triples = []
        for (h,r,t) in self.all_triples:
            new_h, new_t = self.entity_mapping_dict[h], self.entity_mapping_dict[t]
            new_r = self.relation_mapping_dict[r]
            new_triples = (new_h, new_r, new_t)
            self.all_new_triples.append(new_triples)

        # shuffle triples
        random.shuffle(self.all_new_triples)

        # split and save data
        self.trainset, self.valset, self.testset = self.splitData()
        
        # save dataset to local file
        self.saveData(folderName, savePath)

    @staticmethod
    def triplesToNxGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.Graph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)

        node_labels = {i : i for i in nodes}
        nx.set_node_attributes(graph, node_labels, "entity")
        
        for (h,r,t) in triples:
            graph.add_edges_from([(h,t,{'relation':r})])

        return graph

    @staticmethod
    def triplesToNxDiGraph(triples):
        # note that triples are with no inverse relations
        graph = nx.DiGraph()
        nodes = list(set([h for (h,r,t) in triples] + [t for (h,r,t) in triples]))
        graph.add_nodes_from(nodes)

        node_labels = {i : i for i in nodes}
        nx.set_node_attributes(graph, node_labels, "entity")

        edge_list = []
        for (h,r,t) in triples:
            edge_list.append((h,t,{'relation':r}))
        graph.add_edges_from(edge_list)
        return graph
        
    @staticmethod
    def random_walk_induced_graph_sampling(complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        Sampled_nodes = set([complete_graph.nodes[index_of_first_random_node]['id']])

        iteration   = 1
        growth_size = 2
        check_iters = 100
        nodes_before_t_iter = 0
        curr_node = index_of_first_random_node; logging.info(f'==> curr_node: {curr_node}')
        while len(Sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % check_iters == 0:
                if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                    print(f'==> boost seaching, skip to No.{curr_node} node')
                    curr_node = random.randint(0, nr_nodes - 1)
                nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    @staticmethod
    def multi_starts_random_walk_induced_graph_sampling(complete_graph, nodes_to_sample, num_starts):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)

        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes        = len(complete_graph.nodes())
        start_candidate = [random.randint(0, nr_nodes - 1) for i in range(num_starts)]
        Sampled_nodes   = set()

        for idx, index_of_first_random_node in enumerate(start_candidate):
            Sampled_nodes.add(complete_graph.nodes[index_of_first_random_node]['id'])
            iteration           = 1
            growth_size         = 2
            check_iters         = 100
            nodes_before_t_iter = 0
            target_num = int((idx+1) * nodes_to_sample / num_starts)
            curr_node  = index_of_first_random_node

            while len(Sampled_nodes) < target_num:
                edges = [n for n in complete_graph.neighbors(curr_node)]
                index_of_edge = random.randint(0, len(edges) - 1)
                chosen_node = edges[index_of_edge]
                Sampled_nodes.add(complete_graph.nodes[chosen_node]['id'])
                curr_node = chosen_node
                iteration = iteration + 1

                if iteration % check_iters == 0:
                    if ((len(Sampled_nodes) - nodes_before_t_iter) < growth_size):
                        print(f'==> boost seaching, skip to No.{curr_node} node')
                        curr_node = random.randint(0, nr_nodes - 1)
                    nodes_before_t_iter = len(Sampled_nodes)

        return Sampled_nodes

    @staticmethod
    def Genpeel_graph_sampling(complete_graph, nodes_to_sample, p=1):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', 'node_name')
        eps = 1e-9
        class Node():
            def __init__(self, id, val):
                self.id = id
                self.val = val
            def __lt__(self, other):
                return True if self.val < other.val else False

        def getdelta(G, n):
            if p < 0:
                return G.degree(n)
            return sum(map(lambda y:y[1]**p - (y[1] - 1)**p, G.degree(G.neighbors(n)))) + G.degree(n)**p

        delta = {}
        heap = []
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n
            delta[n] = getdelta(complete_graph, n)
            heapq.heappush(heap, Node(n, delta[n]))

        nr_nodes = len(complete_graph.nodes())
        lower_bound_nr_nodes_to_sample = nodes_to_sample
        Sampled_nodes = set(range(0, nr_nodes))
        while len(Sampled_nodes) != lower_bound_nr_nodes_to_sample:
            temp = heapq.heappop(heap)
            tempnode, val = temp.id, temp.val
            if abs(val - delta[tempnode]) > eps:
                continue
            Sampled_nodes.remove(tempnode)
            neighbor = list(complete_graph.neighbors(tempnode))
            complete_graph.remove_node(tempnode)
            for i in neighbor:
                if i != tempnode:#
                    delta[i] = getdelta(complete_graph, i)
                    heapq.heappush(heap, Node(i, delta[i]))
            delta[tempnode]=-1

        mini = 100000000
        for n, data in complete_graph.nodes(data=True):
            mini = min(mini, complete_graph.degree(n))
        print('min degree', mini)
        return Sampled_nodes

    def splitData(self):
        '''
            split triples with certain ratio stored in self.split_ratio
        '''
        n1 = int(self.ntriples * self.split_ratio[0])
        n2 = int(n1 + self.ntriples * self.split_ratio[1])
        return self.all_new_triples[:n1], self.all_new_triples[n1:n2], self.all_new_triples[n2:]

    def saveData(self, folderName, savePath):
        if folderName is None:
            if self.mode == 'random_walk':
                if self.repeat_num == 0:
                    if self.payload == 1:
                        folder = 'sampled_{}_{}'.format(args.dataset, self.sample_ratio)
                    else:
                        folder = 'sampled_{}_{}_starts_{}'.format(args.dataset, self.sample_ratio, self.payload)
                else:
                    if self.payload == 1:
                        folder = 'sampled_{}_{}_rp{}'.format(args.dataset, self.sample_ratio, self.repeat_num)
                    else:
                        folder = 'sampled_{}_{}_starts_{}_rp{}'.format(args.dataset, self.sample_ratio, self.payload, self.repeat_num)
            elif self.mode == 'genpeel':
                folder = 'genpeel_{}_{}_p_{}'.format(args.dataset, self.sample_ratio, self.payload)
            elif self.mode == 'littleball':
                raise('littleball should have foldername')
        else:
            folder = folderName
            if self.mode == 'littleball':
                folder = 'littleball_' + folderName

        saveFolder = os.path.join(savePath, folder)
        saveFolder = saveFolder.replace('-', '_')
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        dataDict = {}
        dataDict['nentity']       = self.nentity
        dataDict['nrelation']     = self.nrelation
        dataDict['train_triples'] = self.trainset
        dataDict['valid_triples'] = self.valset
        dataDict['test_triples']  = self.testset
        dataDict['entity_mapping_dict'] = self.entity_mapping_dict
        dataDict['relation_mapping_dict'] = self.relation_mapping_dict
        if 'biokg' in args.dataset: dataDict['entity_dict'] = self.entity_dict

        dictPath = os.path.join(saveFolder, 'dataset.pkl')
        logging.info('==> save to:{}'.format(dictPath))
        pkl.dump(dataDict, open(dictPath, "wb" ))

# 进入
def generate(args_=None):
    if isinstance(args_, dict):
        # args_ = args
        args_.setdefault('dataset', 'wn18rr')
        args_.setdefault('data_path', './dataset/')
        args_.setdefault('sample_ratio', 0.2)
        args_.setdefault('repeat', 0)
        args_.setdefault('num_starts', 10)
        args_.setdefault('perf_file', './results/sampling.txt')
        args_.setdefault('folder_name', None)
        args_.setdefault('genpeel', False)
        args_.setdefault('p', 1)
        args_.setdefault('littleball', False)
        args_.setdefault('random_walk', False)
        
        global args
        args = argparse.Namespace()
        for k, v in args_.items():
            if isinstance(v, str):
                exec('args.{}= "{}" '.format(k, v))
            else:
                exec('args.{}= {} '.format(k, v))
     

    logger = set_logger(args.perf_file)
    logging.info('==> sampled dataset: {}'.format(args.folder_name))
    # 读图
    print('==> loading dataset ...')
    pklFile = os.path.join(args.data_path, args.dataset.replace('-', '_'), 'datasetInfo.pkl')
    print(pklFile)
    # if os.path.exists(pklFile):
    if False:
        datasetInfo = pkl.load(open(pklFile, 'rb')) 
    else:
        args.addInverseRelation = False
        datasetInfo = prepareData(args)
        pkl.dump(datasetInfo, open(pklFile, "wb"))
    print('==> finish loading dataset')


    nentity       = datasetInfo['nentity']
    nrelation     = datasetInfo['nrelation']
    sample_ratio  = args.sample_ratio
    # num_starts    = args.num_starts
    savePath      = args.data_path
    folderName    =args.folder_name
    ori_train_triples = datasetInfo['train_triples']
    train_triples = []
    new_entity_dict = None
    
    
    # tqdm 进度条
    for head, relation, tail in tqdm(ori_train_triples):
        if relation < nrelation:
            train_triples.append((head, relation, tail))

    # clean cache
    del datasetInfo

    # generating the sampled KG
    if args.random_walk:
        generator = KGGenerator('random_walk', args.num_starts, nentity, train_triples, sample_ratio, folderName, savePath, entity_dict=new_entity_dict)
    
    elif args.genpeel:
        generator = KGGenerator('genpeel', args.p, nentity, train_triples, sample_ratio, folderName, savePath, entity_dict=new_entity_dict)

    elif args.littleball:
        generator = KGGenerator('littleball', None, nentity, train_triples, sample_ratio, folderName, savePath, entity_dict=new_entity_dict)

if __name__ == '__main__':
    args = parse_args()

    try:
        args = parse_args()
        if args.random_walk and args.genpeel:
            raise ValueError("args error!")
    except ValueError as e:
        print(e)

    generate()
import argparse
import json
import logging
import sys
import os
import time
import numpy as np
import pickle as pkl
import torch
from ogb.linkproppred import LinkPropPredDataset, Evaluator
from tqdm import tqdm
from utils import *
from KGEModel import *
from config import *
from random_forest import *
from dataloader import *
from base_function import *
import multiprocessing as mp
import setproctitle

def run_model(param, process_no=0):
    return base_run_model(args=args, params_dict=param, process_no=process_no)

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    # data & device setting
    parser.add_argument('-dataset', type=str, default='ICEWS14', help='dataset name')
    parser.add_argument('-data_path', type=str, default='./dataset/')
    parser.add_argument('-model', default='TNTComplEx', type=str)
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int)
    parser.add_argument('-gpu', nargs='+', type=int, default=[])
    parser.add_argument('-earlyStop',  action='store_true')
    # training process setting
    parser.add_argument('-evaluate', action='store_true', help='evaluate mode')
    parser.add_argument('-search', action='store_true', help='search hyper-parameters')
    parser.add_argument('-no_sort', default=False, action='store_true') ##
    parser.add_argument('-shuffle_sort', default=False, action='store_true') ##
    parser.add_argument('-max_threads', default=1, type=int) ##
    parser.add_argument('-saveRegressor', default=False, action='store_true') ##
    parser.add_argument('-randomSearch', default=False, action='store_true') ##
    parser.add_argument('-resume', default=False, action='store_true', help='resume training or not')
    parser.add_argument('-saveEmbedding', action='store_true', help='save embedding to local files')#saveEmbedding
    parser.add_argument('-loadPretrain', action='store_true', help='load pretrain parameters or not')
    parser.add_argument('-pretrainPath', type=str, default=None)
    parser.add_argument('-eval_test',  action='store_true')
    parser.add_argument('-test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('-evaluate_times', default=1, type=int, help='repeat evaluation times')
    parser.add_argument('-max_steps', default=100000, type=int)
    parser.add_argument('-valid_steps', default=2500, type=int)
    parser.add_argument('-max_trials', default=200, type=int)
    parser.add_argument('-HPO_msg',  default='', type=str)
    parser.add_argument('-seed', default=1, type=int)
    parser.add_argument('-pretrain_dataset', default=None, type=str)
    parser.add_argument('-topNumToStore', default=10, type=int)  # for pretraining 
    parser.add_argument('-HPO_acq', default='BORE', type=str)  # ['max', 'EI', 'UCB', 'BORE']
    parser.add_argument('-space', default='full', type=str)
    
    return parser.parse_args(args)

def set_logger(log_file):
    '''
    save logs to checkpoint and console
    DEBUG INFO WARNING ERROR CRITICAL
    '''
    # TODO: logging.Filter: filter string which contents 'RGCNLayer'
    class stringFilter(logging.Filter):
        def filter(self, record):
            return not 'RGCNLayer' in record.getMessage()


    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)  
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    for h in logger.handlers:
        h.setFormatter(formatter)
    logger.addFilter(stringFilter())

    return logger

def main(args_=None, logger_flag=True):
    if isinstance(args_, dict):
        # args_ = args
        args_.setdefault('dataset', 'ICEWS14')
        args_.setdefault('data_path', './dataset/')
        args_.setdefault('model', 'TNTComplEx')
        args_.setdefault('cpu', 2)
        args_.setdefault('gpu', [])
        args_.setdefault('earlyStop', False)
        args_.setdefault('evaluate', False)
        args_.setdefault('search', False)
        args_.setdefault('saveRegressor', False) ##
        args_.setdefault('randomSearch', False) ##
        args_.setdefault('no_sort', False) ##
        args_.setdefault('shuffle_sort', False) ##
        args_.setdefault('max_threads', 1)  ##
        args_.setdefault('resume', False)  ##
        args_.setdefault('saveEmbedding', False)
        args_.setdefault('loadPretrain', False)
        args_.setdefault('pretrainPath', None)
        args_.setdefault('eval_test', False)
        args_.setdefault('test_batch_size', 16)
        args_.setdefault('evaluate_times', 1)
        args_.setdefault('max_steps', 100000)
        args_.setdefault('valid_steps', 2500)
        args_.setdefault('max_trials', 200)
        args_.setdefault('HPO_msg', '')
        args_.setdefault('seed', 1)
        args_.setdefault('pretrain_dataset', None)
        args_.setdefault('topNumToStore', 10)
        args_.setdefault('HPO_acq', 'BORE')
        args_.setdefault('space', 'full')

        global args
        args = argparse.Namespace()
        for k, v in args_.items():
            if isinstance(v, str):
                exec('args.{}= "{}" '.format(k, v))
            else:
                exec('args.{}= {} '.format(k, v))

    import time
    cpu_num = 6
    torch.set_num_threads(cpu_num)
    args.gpu  = args.gpu[0] if args.gpu != [] else select_gpu()
    args.date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    args.HPO_msg += args.date


    logging.info('==> using No.{} GPU'.format(args.gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    torch.autograd.set_detect_anomaly(True)
    # check 
    if args.search:
        run_mode = 'search' 
    elif args.evaluate:
        run_mode = 'evaluate'
    else:
        logging.error('==> [Error]: you need to select a mode in "search" or "evaluate"')
        exit()
    
    # check path
    save_paths = ['results', os.path.join('results', '{}_{}'.format(args.dataset, args.model)), os.path.join('results', '{}_{}'.format(args.dataset, args.model), 'saveEmb')]
    for save_path in save_paths:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    args.perf_file = os.path.join('results', '{}_{}'.format(args.dataset, args.model),  'search_log.txt')
    args.perf_dict = os.path.join('results', '{}_{}'.format(args.dataset, args.model),  'search_log.pkl') # fix to this file
    if logger_flag:
        logger = set_logger(args.perf_file)
    
    # load data
    logging.info('preparing training data ...')
    pklFile = os.path.join(args.data_path, args.dataset.replace('-', '_'), 'datasetInfo.pkl')
    # if os.path.exists(pklFile):
    if False: ### TODO
        args.datasetInfo = savePickleReader(pklFile)
    else:
        args.addInverseRelation = True
        args.datasetInfo = prepareData(args)
        pkl.dump(args.datasetInfo, open(pklFile, "wb"))
    print('dataset:', args.datasetInfo['datasetName'], 'nentity', args.datasetInfo['nentity'], 'nrelation', args.datasetInfo['nrelation'])
    logging.info('finish preparing')

    if args.evaluate:
        args.eval_test = True
        args.resume    = True
        exec('eval_params = global_cfg.{}_cfg.eval_params'.format(args.model))
        params = locals()['eval_params'][args.dataset]
        logging.info('Evaluation params for {}: {}'.format(args.model, params))

        test_mrr = []; val_mrr  = []
        for time in range(1, args.evaluate_times+1):
            
            # torch.manual_seed(time)
            # random.seed(time)
            # np.random.seed(time)

            # train from scratch
            eval_result = run_model(params)
            # print(eval_result)
            assert eval_result['status'] == 'OK'

            # calculate metrics
            val_mrr.append( eval_result['val_mrr']) 
            test_mrr.append(eval_result['test_mrr'])
            logging.info(f'==> val_mrr  list: {str(val_mrr)}')
            logging.info(f'==> test_mrr list: {str(test_mrr)}')
            
            if time > 0:
                logging.info('==> eval times={}, test mrr: mean={}, std={}'.format(time, np.mean(test_mrr), np.std(test_mrr)))
                logging.info('==> eval times={}, val  mrr: mean={}, std={}'.format(time, np.mean(val_mrr),  np.std(val_mrr)))

    elif args.search:
        args.HPO_trials = 0
        sample_num      = 1e4
        # meta_feature    = pkl.load(open('./dataset/graph_meta_features.pkl', 'rb'))
        meta_feature = {}

        # if args.dataset not in meta_feature.keys(): 
        #     meta_feature[args.dataset] = np.array([0 for i in range(9)])

        if args.pretrain_dataset != None:
            topkConfigs, topkValMRR, topkTestMRR = get_all_configs(args.pretrain_dataset, args.model)
            ref_dataset_names = [args.pretrain_dataset for i in range(len(topkValMRR))]
            assert len(topkConfigs) > 0

            if args.pretrain_dataset not in meta_feature.keys(): 
                # meta_feature[args.pretrain_dataset] = np.array([0 for i in range(9)])
                meta_feature[args.pretrain_dataset] = np.array([1]*9)

            # convert config to correct format (lr, embedding_range, regu_weight) invert log
            for cfg in topkConfigs:
                if cfg['lr'] > 0:               cfg['lr']              = np.log10(cfg['lr'])
                if cfg['embedding_range'] > 0:  cfg['embedding_range'] = np.log10(cfg['embedding_range']) 
                if cfg['regu_weight'] > 0:      cfg['regu_weight']     = np.log10(cfg['regu_weight'])

            topkConfigs = [reviseConfigViaTrainingMode(cfg) for cfg in topkConfigs]

        assert args.space in ['full', 'reduced']
        assert args.HPO_acq in ['max', 'EI', 'UCB', 'BORE']
        if args.space == 'full':    
            selected_space = global_cfg.full_space

            # update 'dim' search range w.r.t. dataset and model
            if args.dataset == 'ogbl-wikikg2':
                selected_space['dim'] = ('choice', [100])
            else:
                if args.model == 'TuckER':
                    selected_space['dim'] = ('choice', [200, 500])
                elif args.model == 'RESCAL':
                    selected_space['dim'] = ('choice', [500, 1000])

        if args.space == 'reduced':  
            selected_space = global_cfg.reduced_space
        
        if args.saveRegressor: # 如果用了regressor，就不用meta feature了
                regressorPath = os.path.join('results', '{}_{}'.format(args.dataset, args.model),  'regressor.pkl')
                if not os.path.exists(regressorPath):
                    current_stage_data = None
                else:
                    current_stage_data = pkl.load(open(regressorPath, 'rb')) 
        else:
            current_stage_data = None

        # searching 
        if args.pretrain_dataset != None:
            HPO_instance = RF_HPO(kgeModelName=args.model, obj_function=run_model, 
                                    dataset_name=args.dataset, HP_info=selected_space, acq=args.HPO_acq,
                                    # meta_feature=meta_feature[args.dataset],
                                    meta_feature=None,
                                    msg=args.HPO_msg,args=args, current_stage_data=current_stage_data)
            HPO_instance.pretrain_with_meta_feature(topkConfigs, topkValMRR, ref_dataset_names, meta_feature, topNumToStore=args.topNumToStore)
            result = HPO_instance.runTrials(args.max_trials, sample_num, meta_feature=None)

        else:
            # run without pretrain data | pure exploration
            HPO_instance = RF_HPO(kgeModelName=args.model, obj_function=run_model, 
                                    dataset_name=args.dataset, HP_info=selected_space, acq=args.HPO_acq,
                                    meta_feature=None, msg=args.HPO_msg, args=args,current_stage_data=current_stage_data)
            result = HPO_instance.runTrials(args.max_trials, sample_num)

        if args.saveRegressor:
            regressorPath = os.path.join('results', '{}_{}'.format(args.dataset, args.model),  'regressor.pkl')
            pkl.dump({'X':result['X'],'Y':result['Y']}, open(regressorPath, 'wb')) 
        return result['result_record']

if __name__ == '__main__':
    args = parse_args()
    main(args)
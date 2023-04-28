from run import main, set_logger
from sampling import generate
import uuid
from types import SimpleNamespace
from utils import *

setting={
    'gpu':[4],
    'cpu':2,
    'dataset':'FB15k_237',
    'model':'ComplEx',
    'maxSteps':100000,
    'validSteps':2500,
    'testBatchSize':16,
    'saveRegressor': True,
    'stages':
    [
        {
            'max_trials':32,

            'learnFeatures':False,  # use meta features from pretrained-dataset
            'earlyStop':True,       
            'subgraphRatio':0.125,    # sampled subgraph ratio
            'stepsRation':0.05,      # stopping steps ratio
            'space':'reduced',      # search space
        },
        {
            'topNumToStore':4,      # top k to next stage
            'max_trials':8,

            'learnFeatures':False,
            'earlyStop':True,
            'subgraphRatio':0.25,
            'stepsRation':0.1,
            'space':'full',
        },
        {
            'topNumToStore':1,      # top k to next stage
            'max_trials':2,

            'learnFeatures':True,
            'earlyStop':True,
            'subgraphRatio':1,
            'stepsRation':1,
            'space':'full',
        },
    ]
} 

def KGTuner(setting):
    loop_times = 10
    logger_flag = True
    result_list = []
    model = setting.model
    for _ in range(loop_times):
        print('loop:{}/{}'.format(_ + 1, loop_times))
        total_stages = len(setting.stages)
        if _ == 0 or setting.saveRegressor == False:
            datasets = []
        for i, stage in enumerate(setting['stages']):
            if _ == 0 or setting.saveRegressor == False:
                if stage['subgraphRatio'] >= 1:
                    dataset = setting['dataset']
                else:
                    dataset = '{}_{}_{}_{}'.format(setting['dataset'], stage['subgraphRatio'], i, str(uuid.uuid4())[:8])
                    generate(dict(dataset=setting['dataset'], sample_ratio=stage['subgraphRatio'], random_walk=True, folder_name=dataset))
                    # generate(dict(dataset=setting['dataset'], sample_ratio=stage['subgraphRatio'], genpeel=True, p=1, folder_name=dataset))
                if len(datasets):
                    pretrain_dataset = datasets[-1]
                else:
                    pretrain_dataset = None
                datasets.append(dataset)
            else:
                if i > 0:
                    pretrain_dataset = datasets[i - 1]
                else:
                    pretrain_dataset = None
                dataset = datasets[i]
                os.remove(os.path.join('results', '{}_{}'.format(dataset, model),  'search_log.pkl'))
            HB=not stage['learnFeatures'] and i!=0
            print(setting['validSteps'], int(setting['maxSteps'] * stage['stepsRation']))

            
            result = main(dict(
                search=True,
                earlyStop=stage['earlyStop'],
                space=stage['space'],
                dataset=dataset,
                pretrain_dataset=pretrain_dataset,
                cpu_num=setting['cpu'],
                gpu=setting['gpu'],
                valid_steps=setting['validSteps'] if int(setting['maxSteps'] * stage['stepsRation']) % setting['validSteps'] == 0 else int(setting['maxSteps'] * stage['stepsRation']),
                max_steps=int(setting['maxSteps'] * stage['stepsRation']),
                model=setting['model'],
                eval_test=True,
                test_batch_size=setting['testBatchSize'],
                # HB=HB,
                max_trials=stage['max_trials'],
                topNumToStore=stage.get('topNumToStore', None),
                saveRegressor=setting['saveRegressor'],
            ), logger_flag=logger_flag)
            logger_flag = False
            if(i + 1 == len(setting['stages'])):
                result_list.extend(result)
                print(result_list)
if __name__ == '__main__':
    KGTuner(ObjDict(setting))
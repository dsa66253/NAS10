# config.py
from joblib import PrintTime
datasetRoot = "../dataset2"
trainDataSetFolder = datasetRoot+"/train"
testDataSetFolder = datasetRoot+"/test"
PRIMITIVES = [
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'conv_9x9',
    'conv_11x11',
    # 'skip_connect'
]

featureMap = {
    "f0":{
        "channel":3,
        "featureMapDim":128
    },
    "f1":{
        "channel":96,
        "featureMapDim":16
    },
    "f2":{
        "channel":256,
        "featureMapDim":16
    },
    "f3":{
        "channel":384,
        "featureMapDim":8
    },
    "f4":{
        "channel":384,
        "featureMapDim":8
    },
    "f5":{
        "channel":256,
        "featureMapDim":4
    },
    "f6":{
        "channel":256,
        "featureMapDim":4
    }
}
trainMatrix = [
    [[1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1]],
    [[1, 1, 1, 1, 1]],
]
featureMapDim = [
    3,
    96,
    256,
    384,
    384,
    256,
    256,
]
seed = {
    "0": 10,
    "1": 255,
    "2": 830,
    "3": 1830,
    "4": 2830,
    "5": 3830,
    "6": 4830,
    "7": 5830,
    "8": 6830,
    "9": 7830
        
}
cfg_alexnet = {
    'name': 'alexnet',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 128,
    'ngpu': 4,
    'epoch': 150,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64, 
    "cuddbenchMark": False,
    "numOfClasses": 10,
}
# 
cfg_nasmodel = {
    'name': 'NasModel',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 128,
    'start_train_nas_epoch': 4,
    'ngpu': 1,
    'epoch': 45,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64,
    "numOfClasses": 10,
    "numOfLayers": len(trainMatrix),
    "numOfInnerCell": len(trainMatrix[0]),
    "numOfOperations": len(PRIMITIVES),
    "cuddbenchMark": False,
    "numOfKth": len(seed)
    
}

cfg_newnasmodel = {
    'name': 'NewNasModel',
    'clip': False,
    'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 64,
    'ngpu': 1,
    'epoch': 45,
    'decay1': 70,
    'decay2': 90,
    'image_size': 128,
    'pretrain': False,
    'in_channel': 8,
    'out_channel': 64,
    "numOfClasses": 10,
    "numOfLayers": len(trainMatrix),
    "numOfInnerCell": len(trainMatrix[0]),
    "numOfOperations": len(PRIMITIVES),
    "cuddbenchMark": False,
    "numOfKth": len(seed)
}

folder = {
    # "nasSavedModel": "./nasSavedModel",
    # "tensorboard_pdarts_nodrop": "./tensorboard_pdarts_nodrop",
    "savedCheckPoint": "./savedCheckPoint",
    "saved_mask_per_epoch": "./saved_mask_per_epoch",
    "decode_folder": "./weights_pdarts_nodrop",
    # "tensorboard_retrain_pdarts" :"./tensorboard_retrain_pdarts",
    "alpha_pdart_nodrop": "./alpha_pdart_nodrop",
    # "weights_retrain_pdarts": "./weights_retrain_pdarts",
    "retrainSavedModel": "./retrainSavedModel",
    "pltSavedDir": "./plot",
    "accLossDir": "./accLoss",
    "log": "./log",
    "tensorboard_trainNas": "./tensorboard_trainNas",
    "tensorboard_retrain": "./tensorboard_retrain",
    "decode": "./decode"
    
}


epoch_to_drop = [10, 25, 35] #????????????epoch?????????????????????
dropNum = [1, 1, 1] #?????????epoch??????1?????????alpha?????????



PRIMITIVES_max = [
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'conv_1x1',
    'max_pool_3x3',
    'avg_pool_3x3'
]

PRIMITIVES_skip = [
    'conv_3x3',
    'conv_5x5',
    'conv_7x7',
    'conv_9x9',
    'conv_11x11',
    'skip_connect'
]

import shutil
import os


deleteDirList = ["./nasSavedModel",
                "./tensorboard_pdarts_nodrop",
                "./savedCheckPoint",
                "./saved_mask_per_epoch",
                "./weights_pdarts_nodrop",
                "./tensorboard_retrain_pdarts",
                "./alpha_pdart_nodrop",
                "./weights_retrain_pdarts",
                "./retrainSavedModel",
                "./plot",
                "./accLoss",
                "./tensorboard_trainNas",
                "./tensorboard_retrain",
                "./decode"        
                ]

for folder in deleteDirList:
    if os.path.exists(folder):
        shutil.rmtree(folder)
import os
os.environ["PYTORCH_JIT"] = "0" #DO NOT DEL THIS LINE

import sys

import torch

#from models import Effnet_MMC
from dataset1 import get_df, get_transforms, MMC_FPDataset
import livedet_func
#from utils.util import *
from tqdm import tqdm
import pandas as pd # <- for ensemble df

import matplotlib.pylab as plt

#Argument len check
if (len(sys.argv)-1) != 5 :
    print('\n ====== EXE ARGUMENT ERROR =====')
    print('WRONG .EXE ARGUMENT, PLEASE CHECK YOUR ARGUMENT (INPUT OF ARGUMENT LENGTH : {})'.format(len(sys.argv)-1))
    sys.exit(1)

class Arguments:
    def __init__(self, ndataset, templateimagesfile, probeimagesfile, livenessoutputfile, IMSoutputfile):
        self.ndataset           = ndataset
        self.templateimagesfile = templateimagesfile
        self.probeimagesfile    = probeimagesfile
        self.livenessoutputfile = livenessoutputfile
        self.IMSoutputfile      = IMSoutputfile

args = Arguments(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])

def main():
    #Get dataframe template & probe
    df_template, template_targetind = get_df(args.templateimagesfile)
    df_probe, probe_target_ind = get_df(args.probeimagesfile)

    #Get Albumentaion transforms template & probe (imagesize : 256)
    transforms_template, transforms_probe = get_transforms(256)

    print(df_template)
    print(df_probe)

    #Get Dataset
    dataset_fp = MMC_FPDataset(df_template, df_probe, transforms_template, transforms_probe)

    #Get test_loader
    test_loader = torch.utils.data.DataLoader(dataset_fp, batch_size=4, num_workers=0, shuffle=False)

    #### liveness ####
    liveness_score_df = livedet_func.get_fp_livenessScore(test_loader, model_mode=args.ndataset)
    print(liveness_score_df)

    ###### matcher #####
    # livedet_func.get_fp_matchingScore(test_loader)
    

        


    


if __name__ == '__main__':
    main()

    


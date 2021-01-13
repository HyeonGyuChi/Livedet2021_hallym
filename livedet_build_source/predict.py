import os
os.environ["PYTORCH_JIT"] = "0" #DO NOT DEL THIS LINE

import sys

# from models import Effnet_MMC, Resnest_MMC, Seresnext_MMC
from models import Effnet_MMC
from dataset1 import get_df_stone, get_transforms, MMC_ClassificationDataset
from utils.util import *
from tqdm import tqdm
import pandas as pd # <- for ensemble df

# def parse_args -> class
# Can be used as dictionary (ClassInstance.__dict__)
class ModelInfo:
    def __init__(self):
        self.kernel_type = '10fold_b3_10ep'

        #self.data_dir = './data/'
        #self.data_folder = './data/'

        self.image_size = 456
        self.enet_type = 'tf_efficientnet_b3_ns' #tf_efficientnet_b3_ns
        self.batch_size = 4
        self.num_workers = 0
        self.out_dim = 2
        self.use_amp = True
        self.use_ext = False # <- True
        self.k_fold = 10

        self.use_meta = False #True
        self.DEBUG = False # <- False (smaple 개수에 따라 잘 설정) (df_test)를 shffle하는 용도 (IMPORTANT args.DEBUG == False ALWAYS)
        self.model_dir = 'weights'
        self.log_dir = './logs'
        self.sub_dir = './subs'
        self.eval = "best" #choices=['best', 'best_no_ext', 'final'] <- 사용 안될것.
        self.n_test = 1 # TTA 횟수 1일경우 실행 x
        self.CUDA_VISIBLE_DEVICES = '0'
        self.n_meta_dim = '512,128'

        
        #Get argv[1,2,3]
        self.model_mode = sys.argv[1]
        self.source_dir = sys.argv[2]
        self.output_dir = sys.argv[3]


#Argument check
if (len(sys.argv)-1) != 3 :
    print('\n ====== EXE ARGUMENT ERROR =====')
    print('WRONG .EXE ARGUMENT, PLEASE CHECK YOUR ARGUMENT (INPUT OF ARGUMENT LENGTH : {})'.format(len(sys.argv)-1))
    sys.exit(1)


#IMPORTANT : Get ModelInfo HERE!!!!!!
args = ModelInfo()


def main():

    #Get test dataframe & target_idx
    df_test, meta_features, n_meta_features, target_idx = get_df_stone(
        k_fold = args.k_fold,
        source_path = args.source_dir,
        out_dim = args.out_dim,
        use_meta = args.use_meta,
        use_ext = args.use_ext
    )


    #Get validation transforms 
    transforms_val = get_transforms(args.image_size)


    #IMPORTANT : args.DEBUG == False ALWAYS!!!
    if args.DEBUG: 
        # batch_size 배수만큼 (testdatset 개수에 따라 주의) -> shuffle # 추후 앙상블 할 때 조심, 같은 데이터가 아닌 서로 다른 데이터 결과끼리 앙상블 될 가능성
        df_test = df_test.sample(args.batch_size * 2)


    #Print logs
    print('--Get Model Info--')
    print('==== df_test ====')
    print(df_test)
    print('==== meta_features ====')
    print(meta_features)
    print('==== n_meta_features ====')
    print(n_meta_features)
    print('==== target_idx ====')
    print(target_idx)
    print('==== transforms_val ====')
    print(transforms_val)



    dataset_test = MMC_ClassificationDataset(df_test, 'test', meta_features, transform=transforms_val)
    # shuffle = false되도록 변경
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


    ############
    
    # from livedet_func.py 

    ############


    PROBS = []
    folds = range(args.k_fold) # fold 개수만큼 반복
    df_avg = pd.DataFrame() # for ensemble df


    #Loop 10 fold
    for fold in folds:

        # model mode 체크  1 = (Dermalog) 3 = (Greenbit) 
        if args.model_mode == '1' :
            model_file = os.path.join(args.model_dir, f'Dermalog_best_fold{fold}.pth')
        elif args.model_mode == '3' :
            model_file = os.path.join(args.model_dir, f'GreenBit_best_fold{fold}.pth')
    
        ''' 기존 args.eval code freeze
        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
            print('eval_best mode')
        '''

        # Get model
        model = ModelClass(
            args.enet_type,                                             # tf_efficientnet_b3_ns
            n_meta_features=n_meta_features,                            # 0
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],  # '512,128' ?
            out_dim=args.out_dim                                        # only 2
        )


        # load model on CPU or GPU
        model = model.to(device)

        
        #load .pth file
        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
            print(f'\n MODEL_PT_LOADED!!! -> {model_file}')
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)


        #Start model eval()
        model.eval()


        ##########################################################################################
        PROBS = []
        with torch.no_grad():
            for (data) in tqdm(test_loader):

                if args.use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
                else: # 여기만 실행
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for I in range(args.n_test): # tta 횟수 (I) n_test=1이면 tta실행 x
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

                probs /= args.n_test

                PROBS.append(probs.detach().cpu())

        PROBS = torch.cat(PROBS).numpy()
        ##########################################################################################


        ### Ensemble ###
        df_test['target'] = PROBS[:, target_idx] # target_idx = 0 이므로 label이 0으로 부여된 확률만 가져옴
        df_avg[str(fold)] = PROBS[:, target_idx]

    df_sum = df_avg.sum(axis=1) / args.k_fold
    change_txt = open(args.output_dir,'w')
    

    #Write result on file(.txt)
    for i in df_sum:
        change_txt.write(str(int(float(i) * 100)) + '\n')

    change_txt.close()

if __name__ == '__main__':
    ## Need some try-catch to argv len

    # os.makedirs(args.sub_dir, exist_ok=True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    
    # model_mode error처리
    if args.model_mode not in ('1', '3') :
        print('\n ====== MODEL_MODE ERROR =====')
        print('You input wrong MODEL_MODE (INPUT OF MODEL_MODE : {})'.format(args.model_mode))
        print('Please input MODEL_MODE only [1,3] == [Greenbit, Dermalog]')
        sys.exit(1)


    # source_dir error리턴
    if not os.path.exists(args.source_dir) :
        print('\n ====== SOURCE_DIRECTORY ERROR =====')
        print('SOURCE FILE is not exist')
        print('Please check your SOURCE DIRECTORY (INPUT OF SOURCE DIR : {})'.format(args.source_dir))
        sys.exit(1)

    
    #Get model type
    if 'efficientnet' in args.enet_type:
        ModelClass = Effnet_MMC
    # elif args.enet_type == 'resnest101':
    #     ModelClass = Resnest_MMC
    #     print('resnest101')
    # elif args.enet_type == 'seresnext101':
    #     ModelClass = Seresnext_MMC
    #     print('seresnext101')
    else:
        raise NotImplementedError()

    #Print log for ModelInfo
    print('==== args.dict ====')
    print(args.__dict__)


    # device 설정 -- CUDA 사용시 out of memory error 발생가능
    if torch.cuda.is_available() :
        print(' === USING CUDA === ')
        device = torch.device('cuda')
    else :
        print('=== USING CPU ===')
        device = torch.device('cpu')

    main()


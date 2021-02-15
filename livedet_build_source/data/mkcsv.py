import os
import glob
import pandas as pd

# Options
###############################
'''
DATA_PATH : Data 경로

target_scanner
    - 1 : GreenBit
    - 3 : Dermalog

_shuffle
    - 0 : 셔플 없음
    - 1 : 모두 섞음
    - 2 : 짝수만 섞음

no_fake
    - True : Fake 이미지 포함 안함
    - False : Fake 이미지 포함
'''
DATA_PATH = "C:\\Users\\lab\\Documents\\Github\\Livedet2021\\livedet_build_source\\data"

target_scanner = 1
# target_scanner = 3

#_shuffle = 0
#_shuffle = 1
_shuffle = 2
#_shuffle = 3

no_fake = True
###############################

GREENBIT_PATH = "livdet2021train\GreenBit"
DERMALOG_PATH = "livdet2021train\Dermalog"

target_path = ""
if target_scanner == 1:
    target_path = GREENBIT_PATH
elif target_scanner == 3:
    target_path = DERMALOG_PATH
else:
    print("target_scanner 딴거 골라요")

###############################GET templateimagesfile.txt################################
f_template = open("templateimagesfile_0214.txt", 'w')

cnt = 0
for file_path in glob.iglob(target_path + "\**\*.png", recursive=True):
    cnt += 1

    if no_fake and file_path.split('\\')[3] == 'Fake':
        continue

    print(DATA_PATH + "\\" + file_path + "\n")
    f_template.write(DATA_PATH + "\\" +file_path + "\n")
print(cnt)

f_template.close()
########################################################################################


###############################GET probeimagesfile.txt################################
f_probe = open("probeimagesfile_0214.txt", 'w')

cnt = 0
for file_path in glob.iglob(target_path + "\**\*.png", recursive=True):
    cnt += 1

    if no_fake and file_path.split('\\')[3] == 'Fake':
        continue    

    print(DATA_PATH + "\\" + file_path + "\n")
    f_probe.write(DATA_PATH + "\\" + file_path + "\n")
print(cnt)

f_probe.close()
########################################################################################

if _shuffle == 0 :
    pass
elif _shuffle == 1:
    ################################Shuffle1################################
    f_shuffle = pd.read_csv(DATA_PATH + "\\" + "probeimagesfile_0214.txt", delimiter='\n', header=None)
    f_shuffle = f_shuffle.sample(frac=1, random_state=3)

    print(f_shuffle)

    f_shuffle.to_csv("probeimagesfile_0214.txt", index=False, header=None)
    #######################################################################
elif _shuffle == 2:
    ################################Shuffle2################################
    f_shuffle = pd.read_csv(DATA_PATH + "\\" + "probeimagesfile_0214.txt", delimiter='\n', header=None)
    f_shuffle.iloc[::2] = f_shuffle.iloc[::2].sample(frac=1, random_state=3)

    print(f_shuffle)

    f_shuffle.to_csv("probeimagesfile_0214.txt", index=False, header=None)
    #######################################################################
elif _shuffle == 3:
    f_template = pd.read_csv(DATA_PATH + "\\" + "templateimagesfile_0214.txt", delimiter='\n', header=None)
    f_shuffle = pd.read_csv(DATA_PATH + "\\" + "probeimagesfile_0214.txt", delimiter='\n', header=None)
    
    for i in range(f_shuffle.shape[0]):
        _str = f_shuffle.iloc[i][0]
        _str = _str[:-5] + '1.png'
        print(_str)
        f_shuffle.iloc[i][0] = _str

    f_shuffle.to_csv("probeimagesfile_0214.txt", index=False, header=None)

    print(f_shuffle)
print("DONE....")
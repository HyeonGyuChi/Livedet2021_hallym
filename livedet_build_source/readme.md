# LIVEDET 2021 Competition
##  ==== THIS IS NOT FINAL SUBMISSION ====
PLEASE CHECK THIS FILE CAN EXECUTE RIGHTLY ON YOUR SYSTEM
=====


## Hallym_MMC demo v1.0
> This Executable file is for FingerPirnt Liveness Detection Model in Livedet 2021

### Team
|Name|Deparment|Roll|
|------|---|---|
|Jong-Uk How  | Hallym prof. | Team Leader


### Build enviroment
- WINDOWS 10(x64)
- Intel i5-10500 (3.10GHz)
- Geforce RTX 2080 super (driver v461.09)
- CUDA 10.2.89 (cuDNN v8.0.4)
    - *RTX 30.. IS NOT SUPORRTED*
- Python 3.8.4
- requirements.txt
- nvcc.txt

### Checked enviroment
- Intel i9 9900K, GTX 1660 TI (GPU)
- AMD RYZEN 5 3600, RTX 2060 SUPER (GPU)
- Intel i5 4670, GTX 750 (CPU)

### Structure
**<U> PLEASE DON'T REMOVE WEIGHTS FOLDER, JUST KEEP THIS RELATIVE PATH </U>**
```shell
root:.
│  nvcc.txt
│  predict.exe
│  requirements.txt
│
└─weights
        ....pth
```



### How to execute
```shell 
$ predict.exe MODE SOURCE_FILE_PATH OUTPUT_FILE_PATH
```
- MODE : [1, 3]
    - 1 : Greenbit
    - 3 : Dermalog
- SOURCE_FILE_PATH
    - Full path of input file that defined input image file (.txt)
- OUTPUT_FILE_PATH
    - Full path of output file that record Liveness Score(.txt)

### Example
```shell
## Dermalog
$ predict.exe 1 C:/Users/aacl/Desktop/livedet_build/data/dermalog_test.txt C:/Users/aacl/Desktop/livedet_build/output/dermalog_result.txt

## GreenBit
$ predict.exe 3 C:/Users/aacl/Desktop/livedet_build/data/greenbit_test.txt C:/Users/aacl/Desktop/livedet_build/output/greenbit_result.txt
```

> You should write full(absolute) path of image data what you want to predict in SOURCE_FILE_PATH(.txt)
```shell
## Example file of SOURCE_FILE_PATH(.txt)
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_1.png
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_2.png
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_3.png
...

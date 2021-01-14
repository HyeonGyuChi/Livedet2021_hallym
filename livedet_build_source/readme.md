# LIVEDET 2021 Competition
##  ==== THIS IS NOT FINAL SUBMISSION ====
PLEASE CHECK THIS FILE CAN EXECUTE RIGHTLY ON YOUR SYSTEM
=====
> This Executable file is for FingerPirnt Liveness Detection Model in Livedet 2021 : Challenge 1

## Hallym_MMC demo v1.0
- Just Check for excutable build file with pyinstaller
- [nameOfAlgorithm].exe [ndataset] [templateimagesfile] [probeimagesfile] [livenessoutputfile] [IMSoutputfile]
- predict.exe [MODE] [SOURCE_FILE_PATH] [OUTPUT_FILE_PATH]

## Hallym_MMC demo v2.0
- Change Submission form [Satisfy Livedet 2021 Challenge 1 Submission](https://livdet.diee.unica.it/index.php/home/algorithm-specifications)
- [nameOfAlgorithm].exe [ndataset] [templateimagesfile] [probeimagesfile] [livenessoutputfile] [IMSoutputfile]



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

### Checked Excutable enviroment
- Intel i5-10500, - Geforce RTX 2080 super with cuda 10.2.89
- Intel i9 9900K, GTX 1660 TI (GPU) with cuda 10.2.89
- AMD RYZEN 5 3600, RTX 2060 SUPER (GPU) with cuda 10.2.89
- Intel i5 4670, GTX 750 (CPU)

### How to Build ###
1. check your enviroments, if you want to use cuda, should install cuda
2. Install python, check requirements
3. download /livedet_build_souce
4. setting .pth file in /weights (prepare .pth file according to the format)
5. edit predict.spec (site packages, paths)
    - site packages : abs path of python env library
    - paths : abs path of /livedet_build_source
6. use pyinstaller command 
```shell
pyinstaller --onefile --clean predict.spec
```
7. finally, predict.exe file will be made in /dist
8. Relocating .exe file to execute as follows #Structure, #How to execute

------
## Execute Build File v2.0
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
$ predict.exe [ndataset] [templateimagesfile] [probeimagesfile] [livenessoutputfile] [IMSoutputfile]
```
- ndataset : [1, 3]
    - 1 : Greenbit
    - 3 : Dermalog
- templateimagesfile
    - A text file with the list of absolute paths of each template image registered in the system. (.txt)
- probeimagesfile
    - A text file with the list of absolute paths of each image to analyse. (.txt)
- livenessoutputfile
    - The text file with the liveness output of each processed image, in the same order of "probeimagesfile".
- IMSoutputfile
    - The text file with the integrated match score output between the processed image (probe) and the corresponding template.
### Example
```shell
## Greenbit
$ predict.exe 1 C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\templateimagesfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\probeimagesfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\livenessoutputfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\IMSoutputfile.txt

## Dermalog
$ predict.exe 3 C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\templateimagesfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\probeimagesfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\livenessoutputfile.txt C:\Users\lab\Documents\Github\Livedet2021\livedet_build_source\data\IMSoutputfile.txt
```

```shell
## Example file of templateimagesfile(.txt) and probeimagesfile(.txt)
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_1.png
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_2.png
C:\Users\aacl\Desktop\livedet_build\data\testset_dermalog\Dermalog_1_24_0_Fake_LEFT_INDEX_3.png
...

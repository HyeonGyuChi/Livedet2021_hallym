:: 가상환경 실행
:: source livebet_env37/Scripts/activate
:: ./livedet_rtx2080_env/Scripts/activate

:: # predict.py 실행
:: ## Greenbit
:: python predict.py 1 C:/Users/aacl/Desktop/livedet_build/data/greenbit_test.txt C:/Users/aacl/Desktop/livedet_build/output/greenbit_result.txt

:: ## Dermnalog
python predict.py 3 C:/Users/aacl/Desktop/livedet_build/data/dermalog_test.txt C:/Users/aacl/Desktop/livedet_build/output/dermalog_result.txt

:: # predict.exe 실행
:: ## greenbit
:: predict.exe 1 C:/Users/aacl/Desktop/livedet_build/data/greenbit_test.txt C:/Users/aacl/Desktop/livedet_build/output/greenbit_result.txt

:: ## dermalog
:: predict.exe 3 C:/Users/aacl/Desktop/livedet_build/data/dermalog_test.txt C:/Users/aacl/Desktop/livedet_build/output/dermalog_result.txt

:: # pyinstaller
:: pyinstaller --onefile --clean predict.spec

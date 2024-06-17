@echo off
setlocal enabledelayedexpansion

set "model_dir=..\data\trained_models"
set "compressed_dir=..\data\compressed"
set "decompressed_dir=..\data\decompressed"
set "data_dir=..\data\processed_files"
set "logs_dir=..\data\logs_data"
set "original_dir=..\data\files_to_be_compressed"

pip install tqdm

mkdir "%model_dir%"
mkdir "%compressed_dir%"

set "model_file=%model_dir%\%1%\biLSTM.hdf5"
set "log_file=%logs_dir%\%1%\biLSTM.log.csv"

python decompressor.py -output "%1.txt" -model !model_file! -model_name biLSTM -input_file_prefix %1 -batch_size 1000 2>&1 >> !log_file!
goto :eof
)

:continue

@echo off
setlocal enabledelayedexpansion

set "compressed_dir=../data/compressed"
set "original_dir=../data/files_to_be_compressed"

pip install tqdm

for %%f in ("%original_dir%\*") do (
    set "filename=%%~nxf"
    echo %%f
    python compressor.py --input_dir "%%f" --batch_size 512 --gpu_id 0 --prefix "!filename!" --hidden_dim 256 --ffn_dim 4096 --seq_len 8 --learning_rate 1e-3 --vocab_dim 64
)

:continue

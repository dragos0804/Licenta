@echo off
setlocal enabledelayedexpansion

set "compressed_dir=../data/compressed"
set "original_dir=../data/files_to_be_compressed"

pip install tqdm

for %%f in ("%compressed_dir%\*.combined") do (
    set "filename=%%~nxf"
    echo %%f
    python decompressor.py --batch_size 512 --prefix "!filename!" --hidden_dim 256 --ffn_dim 4096 --seq_len 8 --vocab_dim 64
)

:continue

setlocal enabledelayedexpansion

set "model_dir=..\data\trained_models"
set "compressed_dir=..\data\compressed"
set "data_dir=..\data\processed_files"
set "logs_dir=..\data\logs_data"
set "original_dir=..\data\files_to_be_compressed"

pip install tqdm

mkdir "%model_dir%"
mkdir "%compressed_dir%"

for %%f in ("%data_dir%\*.npy") do (
	echo "FIRST FOR CMD" 
    echo %%f
    for %%s in (%%~nf) do set "basename=%%s"
	echo "after for set basename"
    echo !basename!

    mkdir "%model_dir%\!basename!"
	echo "start setting params"
    set "model_file=%model_dir%\!basename!\%1.hdf5"
    set "log_file=%logs_dir%\!basename!\%1.log.csv"
    set "output_dir=%compressed_dir%\!basename!"
    set "recon_file_name=!output_dir!\%1.reconstructed.txt"
    set "params_file=%data_dir%\!basename!.param.json"
	echo "params set. params file:"
    echo !params_file!
    set "output_prefix=!output_dir!\%1.compressed"
    mkdir "!output_dir!"
	echo "output dir made"
	
	echo !recon_file_name!
	echo %original_dir%\!basename!.txt

    fc "!recon_file_name!" "%original_dir%\!basename!.txt"
    set "status=!errorlevel!"

	echo "status:"
    echo !status!

    if !status! equ 0 (
        goto :continue
    ) else (
        echo continuing
    )

    mkdir "%model_dir%\!basename!"
    mkdir "%logs_dir%\!basename!"
	echo "model dir and log dir prepared..."
    echo Starting training ... >> "%log_file%"
    python trainer.py -model_name %1 -d "%%~ff" -name !model_file! -log_file !log_file!

    echo Starting Compression ... >> "%log_file%"
    python compressor.py -data "%%~ff" -data_params !params_file! -model !model_file! -model_name %1 -output !output_prefix! -batch_size 1000 2>&1 >> !log_file!
    python decompressor.py -output !recon_file_name! -model !model_file! -model_name %1 -input_file_prefix !output_prefix! -batch_size 1000 2>&1 >> !log_file!
    fc "!recon_file_name!" "%original_dir%\!basename!.txt" >> !log_file!
    goto :eof
)

:continue

setlocal enabledelayedexpansion

set "data_dir=files_to_be_compressed"
set "processed_dir=processed_files"

mkdir "%processed_dir%"

for %%f in ("%data_dir%\*") do (
    echo filename: %%f
    set "s=%%~nxf"
    set "basename=!s:~0,-4!"
    echo !basename!

    set "output_file=%processed_dir%\!basename!.npy"
    set "param_file=%processed_dir%\!basename!.param.json"

    python parse_new.py -input "%%f" -output "!output_file!" -param_file "!param_file!"
    echo - - - - - 
)

endlocal

setlocal enabledelayedexpansion

set "fasta_dir=fasta_files"
set "data_dir=files_to_be_compressed"

for %%F in (%fasta_dir%\*.fa) do (
    echo filename: %%F
    set "s=%%~nxF"
    set "basename=!s:.fa=!"
    echo !basename!

    set "output_file=%data_dir%\!basename!.txt"
    (for /f "usebackq delims=" %%L in ("%%F") do (
        echo %%L | findstr /v "^>" | set /p "= "
    )) > "%output_file%"

    echo - - - - - 
)

endlocal

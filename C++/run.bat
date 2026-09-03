@echo off
REM MSI Linker - C++ RBF Fitting Tool
REM Usage: run.bat <work_dir> <output_dir> <polarity> [--method pinv|ridge|nnls|compare]
REM   work_dir:   Directory containing .mzML/.mzXML files
REM   output_dir: Output directory for xlsx results
REM   polarity:   positive or negative
REM   --method:   Weight solving method (default: ridge)
REM               pinv    = pseudoinverse (fastest, may have negative weights)
REM               ridge   = ridge regression L2 (most stable, recommended)
REM               nnls    = non-negative least squares (guarantees w >= 0)
REM               compare = run all methods and print MSE comparison
REM
REM Example: run.bat ./my_mzml_files ./results positive --method ridge

if "%~1"=="" (
    echo Usage: run.bat ^<work_dir^> ^<output_dir^> ^<polarity^> [--method pinv^|ridge^|nnls^|compare]
    echo   work_dir:   Directory containing .mzML/.mzXML files
    echo   output_dir: Output directory for xlsx results
    echo   polarity:   positive or negative
    echo   --method:   pinv ^| ridge ^(default^) ^| nnls ^| compare
    echo.
    echo Example: run.bat ./my_mzml_files ./results positive --method ridge
    pause
    exit /b 1
)

set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%msi_linker.exe" %1 %2 %3 "%SCRIPT_DIR%data" %4 %5
pause

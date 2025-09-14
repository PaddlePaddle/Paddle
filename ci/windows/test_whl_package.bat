setlocal enabledelayedexpansion

call :test_whl_package || goto test_whl_package_error
goto:eof

:test_whl_package
echo    ========================================
echo    Step 3. Test pip install whl package ...
echo    ========================================

rem Record the exact size of dll and whl files and save them to disk D
cd %BUILD_DIR%
set dll_file=%cd%\paddle\fluid\pybind\libpaddle.dll
for /F "tokens=1-5" %%a in ('dir "%dll_file%"') do (
    echo "%%e" | findstr  "libpaddle.dll" >nul
    if !errorlevel! equ 0 (
        set dllsize=%%d
        goto dll_break
    )
    echo "%%d" | findstr  "libpaddle.dll" >nul
    if !errorlevel! equ 0 (
        set dllsize=%%c
        goto dll_break
    )
)
:dll_break
echo Windows libpaddle.dll Size: %dllsize% bytes
set dllsize_folder=D:\record\dll_size
if not exist "%dllsize_folder%" (
    mkdir %dllsize_folder%
)
if exist "%dllsize_folder%\%PR_ID%.txt" (
    del "%dllsize_folder%\%PR_ID%.txt"
)
echo %dllsize% > %dllsize_folder%\%PR_ID%.txt

set whl_folder=%cd%\python\dist
for /F "tokens=1-5" %%a in ('dir "%whl_folder%"') do (
    echo "%%e" | findstr  ".whl" >nul
    if !errorlevel! equ 0 (
        set whlsize=%%d
        goto whl_break
    )
    echo "%%d" | findstr  ".whl" >nul
    if !errorlevel! equ 0 (
        set whlsize=%%c
        goto whl_break
    )
)
:whl_break
echo Windows PR whl Size: %whlsize% bytes
echo ipipe_log_param_Windows_PR_whl_Size: %whlsize% bytes
set whlsize_folder=D:\record\whl_size
if not exist "%whlsize_folder%" (
    mkdir %whlsize_folder%
)
if exist "%whlsize_folder%\%PR_ID%.txt" (
    del "%whlsize_folder%\%PR_ID%.txt"
)
echo %whlsize% > %whlsize_folder%\%PR_ID%.txt

dir /s /b python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt

@ECHO ON
call %PYTHON_VENV_ROOT%\Scripts\activate.bat
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu
pip install %PADDLE_WHL_FILE_WIN%
%PYTHON_ROOT%\python.exe -m pip uninstall -y paddlepaddle
%PYTHON_ROOT%\python.exe -m pip uninstall -y paddlepaddle-gpu
%PYTHON_ROOT%\python.exe -m pip install %PADDLE_WHL_FILE_WIN%

if %ERRORLEVEL% NEQ 0 (
    echo pip install whl package failed!
    exit /b 1
)

set CUDA_VISIBLE_DEVICES=0
python %work_dir%\paddle\scripts\installation_validate.py
goto:eof

:test_whl_package_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Test import paddle failed, will exit!
exit /b 1

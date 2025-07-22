setlocal enabledelayedexpansion
call :test_unit || goto test_unit_error
goto:eof

:test_unit
echo    ========================================
echo    Step 4. Running unit tests ...
echo    ========================================

cd %BUILD_DIR%
call "%PYTHON_VENV_ROOT%\Scripts\activate.bat"
pip install -r %work_dir%\python\unittest_py\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo pip install unittest requirements.txt failed!
    exit /b 5
)

for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMddHHmmss'"`) do set start=%%i
set start=%start:~4,10%

set FLAGS_call_stack_level=2
dir %THIRD_PARTY_PATH:/=\%\install\openblas\lib
dir %THIRD_PARTY_PATH:/=\%\install\openblas\bin
dir %THIRD_PARTY_PATH:/=\%\install\zlib\bin
dir %THIRD_PARTY_PATH:/=\%\install\mklml\lib
dir %THIRD_PARTY_PATH:/=\%\install\onednn\lib
dir %THIRD_PARTY_PATH:/=\%\install\warpctc\bin
dir %THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib

set PATH=%THIRD_PARTY_PATH:/=\%\install\openblas\lib;%THIRD_PARTY_PATH:/=\%\install\openblas\bin;^
%THIRD_PARTY_PATH:/=\%\install\zlib\bin;%THIRD_PARTY_PATH:/=\%\install\mklml\lib;^
%THIRD_PARTY_PATH:/=\%\install\onednn\lib;%THIRD_PARTY_PATH:/=\%\install\warpctc\bin;^
%THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib;%THIRD_PARTY_PATH:/=\%\install\paddle2onnx\lib;^
%work_dir%\%BUILD_DIR%\paddle\fluid\inference;%work_dir%\%BUILD_DIR%\paddle\fluid\pybind;%work_dir%\%BUILD_DIR%\paddle\fluid\inference\capi_exp;%work_dir%\%BUILD_DIR%\paddle\ir;^
%PATH%
echo PATH=%PATH%>>%GITHUB_ENV%

REM TODO: make ut find .dll in install\onnxruntime\lib
if "%WITH_ONNXRUNTIME%"=="ON" (
    xcopy %THIRD_PARTY_PATH:/=\%\install\onnxruntime\lib\onnxruntime.dll %work_dir%\%BUILD_DIR%\paddle\fluid\inference\tests\api\ /Y
)

if "%WITH_GPU%"=="ON" (
    call:parallel_test_base_gpu
) else (
    call:parallel_test_base_cpu
)

set error_code=%ERRORLEVEL%

for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMddHHmmss'"`) do set end=%%i
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"

if %error_code% NEQ 0 (
    exit /b 8
) else (
    goto:eof
)

:parallel_test_base_gpu
echo    ========================================
echo    Running GPU unit tests in parallel way ...
echo    ========================================

setlocal enabledelayedexpansion

:: set PATH=C:\Windows\System32;C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%
:: cmd /C nvidia-smi -L
:: if %errorlevel% NEQ 0 exit /b 8
:: for /F %%# in ('cmd /C nvidia-smi -L ^|find "GPU" /C') do set CUDA_DEVICE_COUNT=%%#
set CUDA_DEVICE_COUNT=1

:: For hypothesis tests(onednn op and inference pass), we set use 'ci' profile
set HYPOTHESIS_TEST_PROFILE=ci

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:parallel_test_base_cpu
echo    ========================================
echo    Running CPU unit tests in parallel way ...
echo    ========================================

:: For hypothesis tests(onednn op and inference pass), we set use 'ci' profile
set HYPOTHESIS_TEST_PROFILE=ci
%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE% %PRECISION_TEST% %WITH_GPU%

goto:eof

:test_unit_error
:: echo 8 > %cache_dir%\error_code.txt
:: type %cache_dir%\error_code.txt
echo Running unit tests failed, will exit!
exit /b 8

:timestamp
@ECHO OFF
set start=%~1
set dd=%start:~2,2%
set /a dd=100%dd%%%100
set hh=%start:~4,2%
set /a hh=100%hh%%%100
set nn=%start:~6,2%
set /a nn=100%nn%%%100
set ss=%start:~8,2%
set /a ss=100%ss%%%100
set /a start_sec=dd*86400+hh*3600+nn*60+ss
echo %start_sec%

set end=%~2
set dd=%end:~2,2%
set /a dd=100%dd%%%100
if %start:~0,2% NEQ %end:~0,2% (
    set month_day=0
    for %%i in (01 03 05 07 08 10 12) DO if %%i EQU %start:~0,2% set month_day=31
    for %%i in (04 06 09 11) DO if %%i EQU %start:~0,2% set month_day=30
    for %%i in (02) DO if %%i EQU %start:~0,2% set month_day=28
    set /a dd=%dd%+!month_day!
)
set hh=%end:~4,2%
set /a hh=100%hh%%%100
set nn=%end:~6,2%
set /a nn=100%nn%%%100
set ss=%end:~8,2%
set /a ss=100%ss%%%100
set /a end_secs=dd*86400+hh*3600+nn*60+ss
set /a cost_secs=end_secs-start_sec
echo "Windows %~3 Time: %cost_secs%s"
set tempTaskName=%~3
echo ipipe_log_param_Windows_%tempTaskName: =_%_Time: %cost_secs%s
goto:eof

setlocal enabledelayedexpansion

call "%PYTHON_VENV_ROOT%\Scripts\activate.bat"

if not defined SCCACHE_ROOT set "SCCACHE_ROOT=D:\sccache"
set "PATH=%SCCACHE_ROOT%;%PATH%"
if "%WITH_SCCACHE%"=="ON" (
    cmd /C sccache -V || call :install_sccache

    sccache --stop-server 2> NUL
    del %SCCACHE_ROOT%\sccache_log.txt

    :: Locally storage on windows
    if not exist %SCCACHE_ROOT% mkdir %SCCACHE_ROOT%
    set "SCCACHE_DIR=%SCCACHE_ROOT%\.cache"

    :: Sccache will shut down if a source file takes more than 10 mins to compile
    set SCCACHE_IDLE_TIMEOUT=0
    set SCCACHE_CACHE_SIZE=100G
    set "SCCACHE_ERROR_LOG=%SCCACHE_ROOT%\sccache_log.txt"
    set SCCACHE_LOG=quiet

    @REM :: Distributed storage on windows
    @REM set SCCACHE_ENDPOINT=s3.bj.bcebos.com
    @REM set SCCACHE_BUCKET=paddle-github-action
    @REM set SCCACHE_S3_KEY_PREFIX=sccache/
    @REM set SCCACHE_S3_USE_SSL=true

    sccache --start-server
    sccache -z
    goto :begin_cmake
) else (
    del %SCCACHE_ROOT%\sccache.exe 2> NUL
    goto :begin_cmake
)

:install_sccache
echo There is not sccache in this PC, will install sccache.
echo Download package from https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe
python -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe')"
xcopy sccache.exe %SCCACHE_ROOT%\ /Y
del sccache.exe
goto:eof

:begin_cmake
call :cmake || goto cmake_error
goto :begin_build

:cmake
echo    ========================================
echo    Step 1. Cmake ...
echo    ========================================

mkdir %BUILD_DIR%
rem set vs language to english to block showIncludes, this need vs has installed English language package.
set VSLANG=1033
rem Configure the environment for 64-bit builds. 'DISTUTILS_USE_SDK' indicates that the user has selected the compiler.
if not defined vcvars64_dir set "vcvars64_dir=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat" & echo vcvars64_dir=!vcvars64_dir!>> %GITHUB_ENV%
echo %vcvars64_dir%
call "%vcvars64_dir%"

set DISTUTILS_USE_SDK=1
rem Windows 10 Kit bin dir
set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17763.0\x64;%PATH%"
rem Use 64-bit ToolSet to compile
set PreferredToolArchitecture=x64
echo PreferredToolArchitecture=x64>>%GITHUB_ENV%

for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMddHHmmss'"`) do set start=%%i
set start=%start:~4,10%

if not defined CUDA_TOOLKIT_ROOT_DIR set "CUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2"
set "PATH=%TENSORRT_ROOT:/=\%\lib;%CUDA_TOOLKIT_ROOT_DIR:/=\%\bin;%CUDA_TOOLKIT_ROOT_DIR:/=\%\libnvvp;%PATH%"

if "%WITH_GPU%"=="ON" (
    set cuda_version=%CUDA_TOOLKIT_ROOT_DIR:~-4%
    if "!cuda_version!"=="12.0" (
        set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x64;%PATH%"
    )
)
echo %PATH%

rem CUDA_TOOLKIT_ROOT_DIR in cmake must use / rather than \
set "TENSORRT_ROOT=%TENSORRT_ROOT:\=/%"
set "CUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR:\=/%"

rem install ninja if GENERATOR is Ninja
if "%GENERATOR%" == "Ninja" (
    rem Set the default generator for cmake to Ninja
    setx CMAKE_GENERATOR Ninja
    pip install ninja
    if %errorlevel% NEQ 0 (
        echo pip install ninja failed!
        exit /b 5
    )
)

rem ------show summary of current GPU environment----------
cmake --version
if "%WITH_GPU%"=="ON" (
    nvcc --version
    nvidia-smi 2>NUL
)

rem ------set third_party cache dir------
if "%WITH_TPCACHE%"=="OFF" (
    set THIRD_PARTY_PATH=%work_dir:\=/%/%BUILD_DIR%/third_party
    echo THIRD_PARTY_PATH=%THIRD_PARTY_PATH%>> %GITHUB_ENV%
    goto :cmake_impl
)

rem clear third party cache every ten days
for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMddHHmmss'"`) do set datetime=%%i
set day_now=%datetime:~6,2%
set day_before=-1
echo echo [%cache_dir%]
set /p day_before=< %cache_dir%\day_third_party.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day_third_party.txt
    type %cache_dir%\day_third_party.txt
    if %day_now% EQU 21 (
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 11 (
        rmdir %cache_dir%\third_party /s/q
    )
    if %day_now% EQU 01 (
        rmdir %cache_dir%\third_party /s/q
    )
)

echo set -ex > cache.sh
echo md5_content=$(cat %work_dir:\=/%/cmake/external/*.cmake^|md5sum^|awk '{print $1}')$(git submodule status^|md5sum^|awk '{print $1}')>>cache.sh
echo echo ${md5_content}^>md5.txt>>cache.sh

%cache_dir%\tools\busybox64.exe cat cache.sh
%cache_dir%\tools\busybox64.exe bash cache.sh

set /p md5=< md5.txt
if "%WITH_GPU%"=="ON" (
    set cuda_version=%CUDA_TOOLKIT_ROOT_DIR:~-4%
    set sub_dir=cuda!cuda_version:.=!
) else (
    set sub_dir=cpu
)

cd /d %work_dir%
python -c "import wget;wget.download('https://paddle-github-action.bj.bcebos.com/windows/third_party_code/%sub_dir%/%md5%.tar.zst')"
if !ERRORLEVEL! EQU 0 (
    echo Getting source code of third party : extracting ...
    zstd -d %md5%.tar.zst && tar -xf %md5%.tar
    del %md5%.tar.zst
    if !errorlevel! EQU 0 (
        echo Getting source code of third party : successful
    )
) else (
    git submodule update --init --recursive
    if !errorlevel! EQU 0 (
        set UPLOAD_TP_CODE=ON
    )
)
if "%UPLOAD_TP_CODE%"=="ON" (
    set BCE_FILE=%cache_dir%\bce-python-sdk-new\BosClient.py
    echo Uploading source code of third_party: checking bce ...
    if not exist %cache_dir%\bce-python-sdk-new (
        echo There is no bce in this PC, will install bce.
        cd /d %cache_dir%
        echo Download package from https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz
        python -c "import wget;wget.download('https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz')"
        python -c "import shutil;shutil.unpack_archive('bos_new.tar.gz', extract_dir='./bce-python-sdk-new',format='gztar')"
    )
    python -m pip install pycryptodome
    python -m pip install bce-python-sdk==0.8.74
    if !errorlevel! EQU 0 (
        cd /d %work_dir%
        echo Uploading source code of third party: compressing ...
        tar -cf %md5%.tar ./third_party ./.git/modules && zstd %md5%.tar
        if !errorlevel! EQU 0 (
            echo Uploading source code of third party: uploading ...
            python !BCE_FILE! %md5%.tar.zst paddle-github-action/windows/third_party_code/%sub_dir% 1>nul
            if !errorlevel! EQU 0 (
                echo Upload source code of third party %md5% to bos paddle-github-action/windows/third_party_code/%sub_dir% successfully.
            ) else (
                echo Failed upload source code of third party to bos, reason: upload failed.
            )
        ) else (
            echo Failed upload source code of third party to bos, reason: compress failed.
        )
        del %md5%.tar.zst
    ) else (
        echo Failed upload source code of third party to bos, reason: install bce failed.
    )
)

set THIRD_PARTY_HOME=%cache_dir:\=/%/third_party/%sub_dir%
set THIRD_PARTY_PATH=%THIRD_PARTY_HOME%/%md5%

echo this is a CI-Windows task, will try to reuse bos and local third_party cache both.

:cmake_impl
if "%WITH_TESTING%"=="ON" (
    cd /d %work_dir%\%BUILD_DIR%
    rem whether to run cpp test
    python -m pip install PyGithub
    python %work_dir%\tools\check_only_change_python_files.py
    if exist %work_dir%\%BUILD_DIR%\only_change_python_file.txt set WITH_CPP_TEST=OFF
    echo WITH_CPP_TEST: %WITH_CPP_TEST%
)

cd /d %work_dir%\%BUILD_DIR%
echo cmake .. -G %GENERATOR% --trace-expand -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUDA_ARCH_BIN=%CUDA_ARCH_BIN% -DCUB_PATH=%THIRD_PARTY_HOME%/cub ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DNEW_RELEASE_ALL=%NEW_RELEASE_ALL% -DNEW_RELEASE_PYPI=%NEW_RELEASE_PYPI% ^
-DNEW_RELEASE_JIT=%NEW_RELEASE_JIT% -DWITH_ONNXRUNTIME=%WITH_ONNXRUNTIME% -DWITH_CPP_TEST=%WITH_CPP_TEST% ^
-DWIN_UNITTEST_LEVEL=%WIN_UNITTEST_LEVEL% -DWITH_NIGHTLY_BUILD=%WITH_NIGHTLY_BUILD% -DWITH_PIP_CUDA_LIBRARIES=%WITH_PIP_CUDA_LIBRARIES% ^
-DWITH_SCCACHE=%WITH_SCCACHE% >> %work_dir%\win_cmake.sh

cmake .. -G %GENERATOR% -DCMAKE_BUILD_TYPE=Release -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT="%TENSORRT_ROOT%" -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% -DCUDA_ARCH_NAME=%CUDA_ARCH_NAME% -DCUDA_ARCH_BIN=%CUDA_ARCH_BIN% -DCUB_PATH=%THIRD_PARTY_HOME%/cub ^
-DCUDA_TOOLKIT_ROOT_DIR="%CUDA_TOOLKIT_ROOT_DIR%" -DNEW_RELEASE_ALL=%NEW_RELEASE_ALL% -DNEW_RELEASE_PYPI=%NEW_RELEASE_PYPI% ^
-DNEW_RELEASE_JIT=%NEW_RELEASE_JIT% -DWITH_ONNXRUNTIME=%WITH_ONNXRUNTIME% -DWITH_CPP_TEST=%WITH_CPP_TEST% ^
-DWIN_UNITTEST_LEVEL=%WIN_UNITTEST_LEVEL% -DWITH_NIGHTLY_BUILD=%WITH_NIGHTLY_BUILD% -DWITH_PIP_CUDA_LIBRARIES=%WITH_PIP_CUDA_LIBRARIES% ^
-DWITH_SCCACHE=%WITH_SCCACHE%
goto:eof

:cmake_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Cmake failed, will exit!
exit /b 7

:begin_build
call :build || goto build_error
goto:eof

:build
echo    ========================================
echo    Step 2. Build Paddle ...
echo    ========================================

for /F %%# in ('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).NumberOfLogicalProcessors"') do set /a PARALLEL_PROJECT_COUNT=%%#*4/5
echo "PARALLEL PROJECT COUNT is %PARALLEL_PROJECT_COUNT%"

set build_times=1
set retry_times=1
rem MSbuild will build third_party first to improve compiler stability.
if NOT "%GENERATOR%" == "Ninja" (
    goto :build_tp
) else (
    goto :build_paddle
)

:build_tp
echo Build third_party the %build_times% time:
if "%GENERATOR%" == "Ninja" (
    ninja third_party
) else (
    MSBuild /m /p:PreferredToolArchitecture=x64 /p:Configuration=Release /verbosity:%LOG_LEVEL% third_party.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build third_party failed, will retry!
        goto :build_tp
    )
)
echo Build third_party successfully!

set build_times=1

:build_paddle
:: reset clcache zero stats for collect PR's actual hit rate
rem clcache.exe -z

rem -------clean up environment again-----------
taskkill /f /im cmake.exe /t 2>NUL
taskkill /f /im MSBuild.exe /t 2>NUL
taskkill /f /im cl.exe /t 2>NUL
taskkill /f /im lib.exe /t 2>NUL
taskkill /f /im link.exe /t 2>NUL
taskkill /f /im vctip.exe /t 2>NUL
taskkill /f /im cvtres.exe /t 2>NUL
taskkill /f /im rc.exe /t 2>NUL
taskkill /f /im mspdbsrv.exe /t 2>NUL
taskkill /f /im csc.exe /t 2>NUL
taskkill /f /im nvcc.exe /t 2>NUL
taskkill /f /im cicc.exe /t 2>NUL
taskkill /f /im ptxas.exe /t 2>NUL
taskkill /f /im eager_generator.exe /t 2>NUL
taskkill /f /im eager_legacy_op_function_generator.exe /t 2>NUL
wmic process where name="eager_generator.exe" call terminate 2>NUL
wmic process where name="eager_legacy_op_function_generator.exe" call terminate 2>NUL
wmic process where name="cmake.exe" call terminate 2>NUL
wmic process where name="cvtres.exe" call terminate 2>NUL
wmic process where name="rc.exe" call terminate 2>NUL
wmic process where name="cl.exe" call terminate 2>NUL
wmic process where name="lib.exe" call terminate 2>NUL

if "%WITH_TESTING%"=="ON" (
    for /F "tokens=1 delims= " %%# in ('tasklist ^| findstr /i test') do taskkill /f /im %%# /t
)

echo Build Paddle the %build_times% time:
if "%GENERATOR%" == "Ninja" (
    set > env_vars.txt
    ninja all
) else (
    MSBuild /m:%PARALLEL_PROJECT_COUNT% /p:PreferredToolArchitecture=x64 /p:TrackFileAccess=false /p:Configuration=Release /verbosity:%LOG_LEVEL% ALL_BUILD.vcxproj
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GEQ %retry_times% (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!
echo 0 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo PATH=%PATH%>> %GITHUB_ENV%
echo THIRD_PARTY_HOME=%THIRD_PARTY_HOME%>> %GITHUB_ENV%
echo THIRD_PARTY_PATH=%THIRD_PARTY_PATH%>> %GITHUB_ENV%
echo CUDA_TOOLKIT_ROOT_DIR=%CUDA_TOOLKIT_ROOT_DIR%>> %GITHUB_ENV%

:: ci will collect sccache hit rate
if "%WITH_SCCACHE%"=="ON" (
    call :collect_sccache_hits
)

goto:timesummary

:build_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Build Paddle failed, will exit!
exit /b 7

:timesummary
for /f "usebackq" %%i in (`powershell -NoProfile -Command "Get-Date -Format 'yyyyMMddHHmmss'"`) do set end=%%i
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "Build"
goto:eof

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

:collect_sccache_hits
sccache -s > sccache_summary.txt
echo    ========================================
echo    sccache statistical summary ...
echo    ========================================
type sccache_summary.txt
for /f "tokens=2,3" %%i in ('type sccache_summary.txt ^| findstr "requests hits" ^| findstr /V "executed C/C++ CUDA"') do set %%i=%%j
if %requests% EQU 0 (
    echo "sccache hit rate: 0%"
    echo ipipe_log_param_sccache_Hit_Hate: 0%
) else (
    set /a rate=!hits!*10000/!requests!
    echo "sccache hit rate: !rate:~0,-2!.!rate:~-2!%%"
    echo ipipe_log_param_sccache_Hit_Hate: !rate:~0,-2!.!rate:~-2!%%
)
goto:eof

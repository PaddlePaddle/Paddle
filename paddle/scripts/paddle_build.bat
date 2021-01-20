@ECHO OFF
rem Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.

rem =================================================
rem       Paddle CI Task On Windows Platform
rem =================================================

@ECHO ON
setlocal

rem -------clean up environment-----------
set work_dir=%cd%
set cache_dir=%work_dir:Paddle=cache%
if not exist %cache_dir%\tools (
    git clone https://github.com/zhouwei25/tools.git %cache_dir%\tools
)
taskkill /f /im op_function_generator.exe
wmic process where name="op_function_generator.exe" call terminate

rem ------initialize common variable------
if not defined GENERATOR set GENERATOR="Visual Studio 14 2015 Win64"
if not defined BRANCH set BRANCH=develop
if not defined WITH_TENSORRT set WITH_TENSORRT=ON 
if not defined TENSORRT_ROOT set TENSORRT_ROOT="D:/TensorRT"
if not defined WITH_MKL set WITH_MKL=ON
if not defined WITH_AVX set WITH_AVX=ON
if not defined WITH_TESTING set WITH_TESTING=ON
if not defined WITH_PYTHON set WITH_PYTHON=ON
if not defined ON_INFER set ON_INFER=ON
if not defined WITH_INFERENCE_API_TEST set WITH_INFERENCE_API_TEST=ON
if not defined WITH_STATIC_LIB set WITH_STATIC_LIB=ON
if not defined WITH_CACHE set WITH_CACHE=OFF
if not defined WITH_TPCACHE set WITH_TPCACHE=ON
if not defined WITH_UNITY_BUILD set WITH_UNITY_BUILD=OFF
if not defined INFERENCE_DEMO_INSTALL_DIR set INFERENCE_DEMO_INSTALL_DIR=%cache_dir:\=/%/inference_demo

rem -------set cache build work directory-----------
rmdir build\python /s/q
del build\CMakeCache.txt

: set CI_SKIP_CPP_TEST if only *.py changed
git diff --name-only %BRANCH% | findstr /V "\.py" || set CI_SKIP_CPP_TEST=ON

if "%WITH_CACHE%"=="OFF" (
    rmdir build /s/q
    goto :mkbuild
)

set error_code=0
type %cache_dir%\error_code.txt
: set /p error_code=< %cache_dir%\error_code.txt
if %error_code% NEQ 0 (
    rmdir build /s/q
    goto :mkbuild
)

setlocal enabledelayedexpansion
git show-ref --verify --quiet refs/heads/last_pr
if %ERRORLEVEL% EQU 0 (
    git diff HEAD last_pr --stat --name-only
    git diff HEAD last_pr --stat --name-only | findstr "cmake/[a-zA-Z]*\.cmake CMakeLists.txt"
    git branch -D last_pr
    git branch last_pr
) else (
    rmdir build /s/q
    git branch last_pr
)

:: git diff HEAD origin/develop --stat --name-only
:: git diff HEAD origin/develop --stat --name-only | findstr ".cmake CMakeLists.txt paddle_build.bat"
:: if %ERRORLEVEL% EQU 0 (
::     rmdir build /s/q
:: )

:mkbuild
if not exist build (
    echo Windows build cache FALSE
    set Windows_Build_Cache=FALSE
    mkdir build
) else (
    echo Windows build cache TRUE
    set Windows_Build_Cache=TRUE
)
echo ipipe_log_param_Windows_Build_Cache: %Windows_Build_Cache%
cd /d build
dir .
dir %cache_dir%
dir paddle\fluid\pybind\Release

rem ------initialize the python environment------
if not defined PYTHON_ROOT set PYTHON_ROOT=C:\Python37
set PYTHON_EXECUTABLE=%PYTHON_ROOT%\python.exe
set PATH=%PYTHON_ROOT%;%PYTHON_ROOT%\Scripts;%PATH%

rem ToDo: virtual environment can't be deleted safely, some process not exit when task is canceled
rem Now use system python environment temporarily
rem %PYTHON_EXECUTABLE% -m pip install virtualenv
rem %PYTHON_EXECUTABLE% -m virtualenv paddle_winci
rem call paddle_winci\Scripts\activate.bat

rem ------pre install python requirement----------
where python
where pip
pip install wheel --user
pip install -r %work_dir%\python\requirements.txt --user
pip install -r %work_dir%\python\unittest_py\requirements.txt --user
if %ERRORLEVEL% NEQ 0 (
    echo pip install requirements.txt failed!
    exit /b 7
)

rem ------pre install clcache and init config----------
pip install clcache --user
:: set USE_CLCACHE to enable clcache
set USE_CLCACHE=1
:: In some scenarios, CLCACHE_HARDLINK can save one file copy.
set CLCACHE_HARDLINK=1
:: If it takes more than 1000s to obtain the right to use the cache, an error will be reported
set CLCACHE_OBJECT_CACHE_TIMEOUT_MS=1000000
:: set maximum cache size to 20G
clcache.exe -M 21474836480

rem ------show summary of current environment----------
cmake --version
nvcc --version
where nvidia-smi
nvidia-smi
python %work_dir%\tools\summary_env.py
%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\get_cpu_info.sh

goto :CASE_%1

echo "Usage: paddle_build.bat [OPTION]"
echo "OPTION:"
echo "wincheck_mkl: run Windows MKL/GPU/UnitTest CI tasks on Windows"
echo "wincheck_openbals: run Windows OPENBLAS/CPU CI tasks on Windows"
exit /b 1

:CASE_wincheck_mkl

rem ------initialize cmake variable for mkl------
set WITH_MKL=ON
set WITH_GPU=OFF
set MSVC_STATIC_CRT=ON

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
call :test_unit || goto test_unit_error
call :test_inference || goto test_inference_error
:: call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

:CASE_wincheck_openblas

rem ------initialize cmake variable for openblas------
set WITH_MKL=ON
set WITH_GPU=ON
set MSVC_STATIC_CRT=OFF

call :cmake || goto cmake_error
call :build || goto build_error
call :test_whl_pacakage || goto test_whl_pacakage_error
call :test_unit || goto test_unit_error
call :test_inference || goto test_inference_error
:: call :check_change_of_unittest || goto check_change_of_unittest_error
goto:success

rem "Other configurations are added here"
rem :CASE_wincheck_others
rem call ...

rem ---------------------------------------------------------------------------------------------
:cmake
echo    ========================================
echo    Step 1. Cmake ...
echo    ========================================

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

@ECHO ON
if not defined CUDA_TOOLKIT_ROOT_DIR set CUDA_TOOLKIT_ROOT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0
set PATH=%CUDA_TOOLKIT_ROOT_DIR%\bin;%CUDA_TOOLKIT_ROOT_DIR%\libnvvp;%PATH%
set CUDA_PATH=%CUDA_TOOLKIT_ROOT_DIR%

rem ------set third_party cache dir------

: clear third party cache every once in a while
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set datetime=%%#
set day_now=%datetime:~6,2%
set day_before=-1
set /p day_before=< %cache_dir%\day.txt
if %day_now% NEQ %day_before% (
    echo %day_now% > %cache_dir%\day.txt
    type %cache_dir%\day.txt
    if %day_now% EQU 20 (
        rmdir %cache_dir%\third_party_GPU/ /s/q
        rmdir %cache_dir%\third_party/ /s/q
    )
)

if "%WITH_TPCACHE%"=="OFF" (
    set THIRD_PARTY_PATH=%work_dir:\=/%/build/third_party
    goto :cmake_impl
)

echo set -ex > cache.sh
echo md5_content=$(cat %work_dir:\=/%/cmake/external/*.cmake  ^|md5sum ^| awk '{print $1}') >> cache.sh
echo echo ${md5_content}^>md5.txt >> cache.sh

%cache_dir%\tools\busybox64.exe cat cache.sh
%cache_dir%\tools\busybox64.exe bash cache.sh

set /p md5=< md5.txt
if "%WITH_GPU%"=="ON" (
    set THIRD_PARTY_PATH=%cache_dir:\=/%/third_party_GPU/%md5%
) else (
    set THIRD_PARTY_PATH=%cache_dir:\=/%/third_party/%md5%
)

:cmake_impl
echo cmake .. -G %GENERATOR% -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT=%TENSORRT_ROOT% -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD%

cmake .. -G %GENERATOR% -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT=%TENSORRT_ROOT% -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD%
goto:eof

:cmake_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Cmake failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:build
@ECHO OFF
echo    ========================================
echo    Step 2. Buile Paddle ...
echo    ========================================

for /F %%# in ('wmic cpu get NumberOfLogicalProcessors^|findstr [0-9]') do set /a PARALLEL_PROJECT_COUNT=%%#*2/3
set build_times=1
:build_tp
echo Build third_party the %build_times% time:
msbuild /m /p:Configuration=Release /verbosity:quiet third_party.vcxproj
if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1  
    if %build_times% GTR 2 (
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
clcache.exe -z

echo Build Paddle the %build_times% time:
if "%WITH_CLCACHE%"=="OFF" (
    msbuild /m:%PARALLEL_PROJECT_COUNT% /p:Configuration=Release /verbosity:normal paddle.sln
) else (
    msbuild /m:%PARALLEL_PROJECT_COUNT% /p:TrackFileAccess=false /p:CLToolExe=clcache.exe /p:CLToolPath=%PYTHON_ROOT%\Scripts /p:Configuration=Release /verbosity:normal paddle.sln
)

if %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    if %build_times% GTR 1 (
        exit /b 7
    ) else (
        echo Build Paddle failed, will retry!
        goto :build_paddle
    )
)

echo Build Paddle successfully!
echo 0 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt

:: ci will collect clcache hit rate
goto :collect_clcache_hits

goto:eof

:build_error
echo 7 > %cache_dir%\error_code.txt
type %cache_dir%\error_code.txt
echo Build Paddle failed, will exit!
exit /b 7

rem ---------------------------------------------------------------------------------------------
:test_whl_pacakage
@ECHO OFF
echo    ========================================
echo    Step 3. Test pip install whl package ...
echo    ========================================

setlocal enabledelayedexpansion

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "Build"
tree /F %cd%\paddle_inference_install_dir\paddle
%cache_dir%\tools\busybox64.exe du -h -d 0 -k %cd%\paddle_inference_install_dir\paddle\lib > lib_size.txt
set /p libsize=< lib_size.txt

for /F %%i in ("%libsize%") do (
    set /a libsize_m=%%i/1024
    echo "Windows Paddle_Inference Size: !libsize_m!M"
    echo ipipe_log_param_Windows_Paddle_Inference_Size: !libsize_m!M
)
%cache_dir%\tools\busybox64.exe du -h -d 0 %cd%\python\dist > whl_size.txt
set /p whlsize=< whl_size.txt
for /F %%i in ("%whlsize%") do echo "Windows PR whl Size: %%i"
for /F %%i in ("%whlsize%") do echo ipipe_log_param_Windows_PR_whl_Size: %%i
dir /s /b python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt

@ECHO ON
pip uninstall -y paddlepaddle
pip uninstall -y paddlepaddle-gpu
pip install -U %PADDLE_WHL_FILE_WIN% --user
if %ERRORLEVEL% NEQ 0 (
    call paddle_winci\Scripts\deactivate.bat 2>NUL
    echo pip install whl package failed!
    exit /b 1
)

set CUDA_VISIBLE_DEVICES=0
python %work_dir%\paddle\scripts\installation_validate.py
goto:eof

:test_whl_pacakage_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Test import paddle failed, will exit!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:test_unit
@ECHO ON
echo    ========================================
echo    Step 4. Running unit tests ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set start=%%#
set start=%start:~4,10%

set FLAGS_call_stack_level=2
dir %THIRD_PARTY_PATH:/=\%\install\openblas\lib
dir %THIRD_PARTY_PATH:/=\%\install\openblas\bin
dir %THIRD_PARTY_PATH:/=\%\install\zlib\bin
dir %THIRD_PARTY_PATH:/=\%\install\mklml\lib
dir %THIRD_PARTY_PATH:/=\%\install\mkldnn\bin
dir %THIRD_PARTY_PATH:/=\%\install\warpctc\bin

pip install requests
python %work_dir%\tools\get_quick_disable_lt.py > Output
if %errorlevel%==0 (
    set /p disable_ut_quickly=<Output
    DEL Output
    ) else (
    set disable_ut_quickly=''
)

set PATH=%THIRD_PARTY_PATH:/=\%\install\openblas\lib;%THIRD_PARTY_PATH:/=\%\install\openblas\bin;^
%THIRD_PARTY_PATH:/=\%\install\zlib\bin;%THIRD_PARTY_PATH:/=\%\install\mklml\lib;^
%THIRD_PARTY_PATH:/=\%\install\mkldnn\bin;%THIRD_PARTY_PATH:/=\%\install\warpctc\bin;%PATH%

if "%NIGHTLY_MODE%"=="ON" (
    set nightly_label="()"
    ) else (
    set nightly_label="(RUN_TYPE=NIGHTLY^|RUN_TYPE=DIST:NIGHTLY^|RUN_TYPE=EXCLUSIVE:NIGHTLY)"
    echo    ========================================
    echo    "Unittests with nightly labels  are only run at night"
    echo    ========================================
)

if "%WITH_GPU%"=="ON" (
    goto:parallel_test_base_gpu
) else (
    goto:parallel_test_base_cpu
)

:parallel_test_base_gpu
echo    ========================================
echo    Running GPU unit tests...
echo    ========================================

setlocal enabledelayedexpansion

:: set PATH=C:\Windows\System32;C:\Program Files\NVIDIA Corporation\NVSMI;%PATH%
:: cmd /C nvidia-smi -L
:: if %errorlevel% NEQ 0 exit /b 8
:: for /F %%# in ('cmd /C nvidia-smi -L ^|find "GPU" /C') do set CUDA_DEVICE_COUNT=%%#
set CUDA_DEVICE_COUNT=1

%cache_dir%\tools\busybox64.exe bash %work_dir%\tools\windows\run_unittests.sh %NIGHTLY_MODE%

goto:eof

:parallel_test_base_cpu
echo    ========================================
echo    Running CPU unit tests in parallel way ...
echo    ========================================
ctest.exe -E "(%disable_ut_quickly%)" -LE %nightly_label% --output-on-failure -C Release -j 8 --repeat until-pass:4 after-timeout:4

goto:eof

:test_unit_error
:: echo 8 > %cache_dir%\error_code.txt
:: type %cache_dir%\error_code.txt
for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"
echo Running unit tests failed, will exit!
exit /b 8

rem ---------------------------------------------------------------------------------------------
:test_inference
@ECHO OFF
echo    ========================================
echo    Step 5. Testing fluid library for inference ...
echo    ========================================

for /F %%# in ('wmic os get localdatetime^|findstr 20') do set end=%%#
set end=%end:~4,10%
call :timestamp "%start%" "%end%" "1 card TestCases Total"
call :timestamp "%start%" "%end%" "TestCases Total"

cd %work_dir%\paddle\fluid\inference\api\demo_ci
%cache_dir%\tools\busybox64.exe bash run.sh %work_dir:\=/% %WITH_MKL% %WITH_GPU% %cache_dir:\=/%/inference_demo %TENSORRT_ROOT%/include %TENSORRT_ROOT%/lib %MSVC_STATIC_CRT%
goto:eof

:test_inference_error
::echo 1 > %cache_dir%\error_code.txt
::type %cache_dir%\error_code.txt
echo Testing fluid library for inference failed!
exit /b 1

rem ---------------------------------------------------------------------------------------------
:check_change_of_unittest
@ECHO OFF
echo    ========================================
echo    Step 6. Check whether deleting a unit test ...
echo    ========================================

cd /d %work_dir%\build
echo set -e>  check_change_of_unittest.sh
echo set +x>> check_change_of_unittest.sh
echo GITHUB_API_TOKEN=%GITHUB_API_TOKEN% >>  check_change_of_unittest.sh
echo GIT_PR_ID=%AGILE_PULL_ID% >>  check_change_of_unittest.sh
echo BRANCH=%BRANCH%>>  check_change_of_unittest.sh
echo if [ "${GITHUB_API_TOKEN}" == "" ] ^|^| [ "${GIT_PR_ID}" == "" ];then>> check_change_of_unittest.sh
echo     exit 0 >>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo set -x>> check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of this PR.         >>  check_change_of_unittest.sh
echo     ============================================ >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_PR.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo num=$(awk 'END{print NR}' ${spec_path})>> check_change_of_unittest.sh
echo echo "Windows 1 card TestCases count is $num">> check_change_of_unittest.sh
echo echo ipipe_log_param_Windows_1_Card_TestCases_Count: $num>> check_change_of_unittest.sh
echo UPSTREAM_URL='https://github.com/PaddlePaddle/Paddle'>>  check_change_of_unittest.sh
echo origin_upstream_url=`git remote -v ^| awk '{print $1, $2}' ^| uniq ^| grep upstream ^| awk '{print $2}'`>>  check_change_of_unittest.sh
echo if [ "$origin_upstream_url" == "" ]; then>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo elif [ "$origin_upstream_url" ^!= "$UPSTREAM_URL" ] ^\>>  check_change_of_unittest.sh
echo         ^&^& [ "$origin_upstream_url" ^!= "$UPSTREAM_URL.git" ]; then>>  check_change_of_unittest.sh
echo     git remote remove upstream>>  check_change_of_unittest.sh
echo     git remote add upstream $UPSTREAM_URL.git>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo if [ ! -e "$(pwd)/../.git/refs/remotes/upstream/$BRANCH" ]; then>>  check_change_of_unittest.sh
echo     git fetch upstream $BRANCH # develop is not fetched>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo git checkout -b origin_pr >>  check_change_of_unittest.sh
echo git checkout -f $BRANCH >>  check_change_of_unittest.sh
echo cmake .. -G %GENERATOR% -DWITH_AVX=%WITH_AVX% -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% ^
-DWITH_TESTING=%WITH_TESTING% -DWITH_PYTHON=%WITH_PYTHON% -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% -DON_INFER=%ON_INFER% ^
-DWITH_INFERENCE_API_TEST=%WITH_INFERENCE_API_TEST% -DTHIRD_PARTY_PATH=%THIRD_PARTY_PATH% ^
-DINFERENCE_DEMO_INSTALL_DIR=%INFERENCE_DEMO_INSTALL_DIR% -DWITH_STATIC_LIB=%WITH_STATIC_LIB% ^
-DWITH_TENSORRT=%WITH_TENSORRT% -DTENSORRT_ROOT=%TENSORRT_ROOT% -DMSVC_STATIC_CRT=%MSVC_STATIC_CRT% ^
-DWITH_UNITY_BUILD=%WITH_UNITY_BUILD% >>  check_change_of_unittest.sh
echo cat ^<^<EOF>>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo     Generate unit tests.spec of develop.               >>  check_change_of_unittest.sh
echo     ============================================       >>  check_change_of_unittest.sh
echo EOF>>  check_change_of_unittest.sh
echo spec_path=$(pwd)/UNITTEST_DEV.spec>>  check_change_of_unittest.sh
echo ctest -N ^| awk -F ':' '{print $2}' ^| sed '/^^$/d' ^| sed '$d' ^> ${spec_path}>>  check_change_of_unittest.sh
echo unittest_spec_diff=`python $(pwd)/../tools/diff_unittest.py $(pwd)/UNITTEST_DEV.spec $(pwd)/UNITTEST_PR.spec`>>  check_change_of_unittest.sh
echo if [ "$unittest_spec_diff" ^!= "" ]; then>>  check_change_of_unittest.sh
echo     set +x>> check_change_of_unittest.sh
echo     approval_line=`curl -H "Authorization: token ${GITHUB_API_TOKEN}" https://api.github.com/repos/PaddlePaddle/Paddle/pulls/${GIT_PR_ID}/reviews?per_page=10000`>>  check_change_of_unittest.sh
echo     set -x>> check_change_of_unittest.sh
echo     if [ "$approval_line" ^!= "" ]; then>>  check_change_of_unittest.sh
echo         APPROVALS=`echo ${approval_line} ^|python $(pwd)/../tools/check_pr_approval.py 1 22165420 52485244 6836917`>>  check_change_of_unittest.sh
echo         echo "current pr ${GIT_PR_ID} got approvals: ${APPROVALS}">>  check_change_of_unittest.sh
echo         if [ "${APPROVALS}" == "FALSE" ]; then>>  check_change_of_unittest.sh
echo             echo "************************************"                >>  check_change_of_unittest.sh
echo             echo -e "It is forbidden to disable or delete the unit-test.\n"        >>  check_change_of_unittest.sh
echo             echo -e "If you must delete it temporarily, please add it to[https://github.com/PaddlePaddle/Paddle/wiki/Temporarily-disabled-Unit-Test]."     >>  check_change_of_unittest.sh
echo             echo -e "Then you must have one RD (kolinwei(recommended) or zhouwei25) approval for the deletion of unit-test. \n"                 >>  check_change_of_unittest.sh
echo             echo -e "If you have any problems about deleting unit-test, please read the specification [https://github.com/PaddlePaddle/Paddle/wiki/Deleting-unit-test-is-forbidden]. \n"   >>  check_change_of_unittest.sh
echo             echo -e "Following unit-tests are deleted in this PR: \n ${unittest_spec_diff} \n"     >>  check_change_of_unittest.sh
echo             echo "************************************"                >>  check_change_of_unittest.sh
echo             exit 1 >>  check_change_of_unittest.sh
echo          fi>>  check_change_of_unittest.sh
echo     else>>  check_change_of_unittest.sh
echo          exit 1 >>  check_change_of_unittest.sh
echo     fi>>  check_change_of_unittest.sh
echo fi>>  check_change_of_unittest.sh
echo git checkout -f origin_pr >>  check_change_of_unittest.sh
%cache_dir%\tools\busybox64.exe bash check_change_of_unittest.sh
goto:eof

:check_change_of_unittest_error
exit /b 1


:timestamp
setlocal enabledelayedexpansion
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


:collect_clcache_hits
for /f "tokens=2,4" %%i in ('clcache.exe -s ^| findstr "entries hits"') do set %%i=%%j
if %hits% EQU 0 (
    echo "clcache hit rate: 0%%"
    echo ipipe_log_param_Clcache_Hit_Rate: 0%%
) else (
    set /a rate=%hits%*10000/%entries%
    echo "clcache hit rate: %rate:~0,-2%.%rate:~-2%%%"
    echo ipipe_log_param_Clcache_Hit_Hate: %rate:~0,-2%.%rate:~-2%%%
)
goto:eof


rem ---------------------------------------------------------------------------------------------
:success
echo    ========================================
echo    Clean up environment  at the end ...
echo    ========================================
taskkill /f /im cmake.exe  2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im link.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL
taskkill /f /im cvtres.exe 2>NUL
taskkill /f /im rc.exe 2>NUL
wmic process where name="op_function_generator.exe" call terminate 2>NUL
taskkill /f /im python.exe  2>NUL
echo Windows CI run successfully!
exit /b 0

ENDLOCAL

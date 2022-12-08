@ECHO OFF
SETLOCAL 
set source_path=%1
set PYTHON_DIR=%2
set WITH_GPU=%3
set WITH_MKL=%4
set ON_INFER=%5
set PADDLE_VERSION=%6
set BATDIR=%7
set CUDA_DIR=%8

set RETRY_TIMES=3

set CUDA_DIR_WIN=%CUDA_DIR:/=\%
set PATH=%CUDA_DIR_WIN%\nvvm\bin\;%CUDA_DIR_WIN%\bin;%PATH%


for /f "tokens=1,2,* delims=\\" %%a in ("%PYTHON_DIR%") do (
	set c1=%%a
	set c2=%%b
)
set PYTHONV=%c2%

echo %CUDA_DIR% | findstr 10.0 > NULL
if %errorlevel% == 0 (set PADDLE_VERSION=%PADDLE_VERSION%
set CUDAV=v10.0)
echo %CUDA_DIR% | findstr 9.2 > NULL
if %errorlevel% == 0 (set PADDLE_VERSION=%PADDLE_VERSION%.post97
set CUDAV=v9.2)
echo %CUDA_DIR% | findstr 9.0 > NULL
if %errorlevel% == 0 (set PADDLE_VERSION=%PADDLE_VERSION%.post97
set CUDAV=v9.0)
echo %CUDA_DIR% | findstr 8.0 > NULL
if %errorlevel% == 0 (set PADDLE_VERSION=%PADDLE_VERSION%.post87
set CUDAV=v8.0)
set PLAT=GPU
if "%WITH_GPU%"=="OFF" (
    set PLAT=CPU
    set CUDAV=CPU
)

if "%WITH_MKL%"=="ON" (
    set BLAS=MKL
) else (
    set BLAS=OPEN
)

if "%ON_INFER%"=="ON" (
    goto :INFERENCE_LIBRARY
)

echo "begin to do build noavx ..."

set "dst_path=%source_path%\build_%PYTHONV%_%PLAT%_%BLAS%_%CUDAV%_noavx"

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%
echo Current directory : %cd%

call:rest_env

echo cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=OFF -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR% -DCUDA_ARCH_NAME=All
cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=OFF -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR% -DCUDA_ARCH_NAME=All

set  MSBUILDDISABLENODEREUSE=1

set BUILD_TYPE=NO_AVX
call:Build

REM -------------------------------------------------------------------------

echo "begin to do build avx ..."
set "dst_path=%source_path%\build_%PYTHONV%_%PLAT%_%BLAS%_%CUDAV%"

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%
echo Current directory : %cd%

call:rest_env

echo cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR% -DCUDA_ARCH_NAME=All
cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR% -DCUDA_ARCH_NAME=All

set  MSBUILDDISABLENODEREUSE=1

set BUILD_TYPE=AVX
call:Build

echo BUILD WHL PACKAGE COMPLETE
goto :END
REM -------------------------------------------------------------------------

:INFERENCE_LIBRARY

echo "begin to do build inference library ..."
set "dst_path=%source_path%\build_INFERENCE_LIBRARY_%PLAT%_%BLAS%_%CUDAV%"

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%
echo Current directory : %cd%

call:rest_env

echo cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=OFF -DON_INFER=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR% -DCUDA_ARCH_NAME=All
cmake %dst_path%\..\Paddle -G "Visual Studio 15 2017 Win64" -T host=x64 -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=OFF -DON_INFER=ON  -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_DIR%  -DCUDA_ARCH_NAME=All

set  MSBUILDDISABLENODEREUSE=1

set BUILD_TYPE=INFERENCE LIBRARY
call:Build

echo PACKAGE INFERENCE LIBRARY

mkdir inference_dist
%PYTHON_DIR%\python.exe -c "import shutil;shutil.make_archive('inference_dist/paddle_inference_install_dir', 'zip', root_dir='paddle_inference_install_dir')"
%PYTHON_DIR%\python.exe -c "import shutil;shutil.make_archive('inference_dist/paddle_install_dir', 'zip', root_dir='paddle_install_dir')"

echo BUILD INFERENCE LIBRARY COMPLETE
goto :END


:Rest_env
echo "Reset Build Environment ..."
taskkill /f /im cmake.exe   2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL
goto:eof

:Build
set build_times=1
:build_thirdparty

echo Build %BUILD_TYPE% Third Party Libraries, Round : %build_times%

echo msbuild /m /p:Configuration=Release third_party.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /m /p:Configuration=Release third_party.vcxproj >> build_thirdparty_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
    echo Build %BUILD_TYPE% Third Party Libraries, Round : %build_times% Failed!
    set /a build_times=%build_times%+1

    if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_thirdparty
  )
)

set build_times=1
:build_paddle

echo Build %BUILD_TYPE% Paddle Solutions, Round : %build_times%

echo msbuild /m /p:Configuration=Release paddle.sln ^>^> build_%build_times%.log
msbuild /m /p:Configuration=Release paddle.sln >> build_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
    echo Build %BUILD_TYPE% Paddle Solutions, Round : %build_times% Failed!
    set /a build_times=%build_times%+1

    if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_paddle
  )
)
goto:eof


:FAILURE
echo BUILD FAILED
exit /b 1

:END
echo BUILD SUCCESSFULLY

ENDLOCAL

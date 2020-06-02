@ECHO OFF
SETLOCAL 
set source_path=%1
set PYTHON_DIR=%2
set WITH_GPU=%3
set WITH_MKL=%4
set ON_INFER=%5
set PADDLE_VERSION=%6
set BATDIR=%7
set release_dir=%8
set CUDA_PATH=%9

set RETRY_TIMES=5


set CUDA_PATH_WIN=%CUDA_PATH:/=\%
set PATH=%CUDA_PATH_WIN%\nvvm\bin\;%CUDA_PATH_WIN%\bin;%PATH%

echo Init Visual Studio Env
call "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

echo Set Net Proxy
set http_proxy=http://172.19.57.45:3128
set https_proxy=http://172.19.57.45:3128

for /f "tokens=1,2,* delims=\\" %%a in ("%PYTHON_DIR%") do (
	set c1=%%a
	set c2=%%b
)
set PYTHONV=%c2%

for /f "tokens=1,2,* delims=/" %%a in ("%CUDA_PATH%") do (
	set x1=%%a
	set x2=%%b
)

set CUDAV=%x2%

if "%WITH_GPU%"=="ON" (
    if "%CUDAV%"=="v8.0" (set PADDLE_VERSION=%PADDLE_VERSION%.post87)
    if "%CUDAV%"=="v9.0" (set PADDLE_VERSION=%PADDLE_VERSION%.post97)
    if "%CUDAV%"=="v9.2" (set PADDLE_VERSION=%PADDLE_VERSION%.post97)
    if "%CUDAV%"=="v10.0" (set PADDLE_VERSION=%PADDLE_VERSION%)
    set PLAT=GPU
) else (
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


set "dst_path=%source_path%\build_%PYTHONV%_%PLAT%_%BLAS%_%CUDAV%_noavx"
echo %dst_path%

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%

echo Reset Build Environment

taskkill /f /im cmake.exe   2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL

set INS=NOAVX
echo "begin to do build noavx ..."

echo Current directory : %cd%

echo cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=OFF -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH% -DCUDA_ARCH_NAME=All
cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=OFF -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH% -DCUDA_ARCH_NAME=All

set  MSBUILDDISABLENODEREUSE=1

set build_times=1
:build_noavx_thirdparty

echo Build NOAVX Third Party Libraries, Round : %build_times%

echo MKL OPTION : %WITH_MKL%

echo msbuild /p:Configuration=Release extern_protobuf.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /p:Configuration=Release extern_protobuf.vcxproj >> build_thirdparty_%build_times%.log

if "%WITH_MKL%"=="ON" (
    echo msbuild /p:Configuration=Release extern_mkldnn.vcxproj ^>^> build_thirdparty_%build_times%.log
    msbuild /p:Configuration=Release extern_mkldnn.vcxproj >> build_thirdparty_%build_times%.log
)

echo msbuild /m /p:Configuration=Release third_party.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /m /p:Configuration=Release third_party.vcxproj >> build_thirdparty_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
    set /a build_times=%build_times%+1
    
    if %build_times% GTR %RETRY_TIMES% (
        goto :FAILURE
    ) else (
        goto :build_noavx_thirdparty
    )
)

set build_times=1
:build_noavx_paddle

echo Build NOAVX Paddle Solution, Round : %build_times%

echo msbuild /m /p:Configuration=Release paddle.sln ^>^> build_%build_times%.log
msbuild /m /p:Configuration=Release paddle.sln >> build_%build_times%.log

IF NOT exist %dst_path%\python\paddle\fluid\core_noavx.pyd  (
    echo  %dst_path%_noavx\python\paddle\fluid\core_noavx.pyd not exist
    set /a build_times=%build_times%+1

    if %build_times% GTR %RETRY_TIMES% (
        goto :FAILURE
    ) else (
        goto :build_noavx_paddle
    )
)

REM -------------------------------------------------------------------------

set "dst_path=%source_path%\build_%PYTHONV%_%PLAT%_%BLAS%_%CUDAV%"
echo %dst_path%

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%

echo Reset Build Environment

taskkill /f /im cmake.exe   2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL

set INS=AVX
echo "begin to do build avx ..."

echo cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH% -DCUDA_ARCH_NAME=All -DNOAVX_CORE_FILE=%dst_path%_noavx\python\paddle\fluid\core_noavx.pyd
cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH% -DCUDA_ARCH_NAME=All -DNOAVX_CORE_FILE=%dst_path%_noavx\python\paddle\fluid\core_noavx.pyd

set  MSBUILDDISABLENODEREUSE=1

set build_times=1
:build_avx_thirdparty

echo Build AVX Third Party Libraries, Round : %build_times%

echo msbuild /p:Configuration=Release extern_protobuf.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /p:Configuration=Release extern_protobuf.vcxproj >> build_thirdparty_%build_times%.log

if "%WITH_MKL%"=="ON" (
    echo msbuild /p:Configuration=Release extern_mkldnn.vcxproj ^>^> build_thirdparty_%build_times%.log
    msbuild /p:Configuration=Release extern_mkldnn.vcxproj >> build_thirdparty_%build_times%.log
)

echo msbuild /m /p:Configuration=Release third_party.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /m /p:Configuration=Release third_party.vcxproj >> build_thirdparty_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
  set /a build_times=%build_times%+1

  if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_avx_thirdparty
  )
)

set build_times=1
:build_avx_paddle

echo Build AVX Paddle Solution, Round : %build_times%

echo msbuild /m /p:Configuration=Release paddle.sln ^>^> build_%build_times%.log
msbuild /m /p:Configuration=Release paddle.sln >> build_%build_times%.log

IF NOT exist %dst_path%\python\dist\*.whl  (
    echo %dst_path%\python\dist\*.whl not exist

   set /a build_times=%build_times%+1

   if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_avx_paddle
  )
)

echo BUILD COMPLETE
goto :END
REM -------------------------------------------------------------------------

:INFERENCE_LIBRARY
set "dst_path=%source_path%\build_INFERENCE_LIBRARY_%PLAT%_%BLAS%_%CUDAV%"
echo %dst_path%

if exist %dst_path% rmdir /q /s %dst_path%
mkdir %dst_path%

cd /d %dst_path%

echo Reset Build Environment

taskkill /f /im cmake.exe   2>NUL
taskkill /f /im msbuild.exe 2>NUL
taskkill /f /im git.exe 2>NUL
taskkill /f /im cl.exe 2>NUL
taskkill /f /im lib.exe 2>NUL
taskkill /f /im git-remote-https.exe 2>NUL
taskkill /f /im vctip.exe 2>NUL

echo "begin to do build inference library ..."

echo cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=OFF -DON_INFER=ON -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH% -DCUDA_ARCH_NAME=All
cmake %dst_path%\..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DCMAKE_BUILD_TYPE=Release -DWITH_PYTHON=OFF -DON_INFER=ON  -DCUDA_TOOLKIT_ROOT_DIR=%CUDA_PATH%  -DCUDA_ARCH_NAME=All

set  MSBUILDDISABLENODEREUSE=1

set build_times=1
:build_thirdparty

echo Build INFERENCE Third Party, Round : %build_times%

echo msbuild /m /p:Configuration=Release third_party.vcxproj ^>^> build_thirdparty_%build_times%.log
msbuild /m /p:Configuration=Release third_party.vcxproj >> build_thirdparty_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
  set /a build_times=%build_times%+1

  if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_thirdparty
  )
)

set build_times=1
:build_inference

echo Build INFERENCE LIBRARY, Round : %build_times%

echo msbuild /m /p:Configuration=Release paddle.sln ^>^> build_%build_times%.log
msbuild /m /p:Configuration=Release paddle.sln >> build_%build_times%.log

IF %ERRORLEVEL% NEQ 0 (
   echo Build INFERENCE LIBRARY, Round : %build_times% Failed!

   set /a build_times=%build_times%+1

   if %build_times% GTR %RETRY_TIMES% (
      goto :FAILURE
  ) else (
      goto :build_inference
  )
)

echo PACKAGE INFERENCE LIBRARY

mkdir inference_dist

"%BATDIR%\7z.exe" a inference_dist\fluid_inference_install_dir.zip fluid_inference_install_dir -r
"%BATDIR%\7z.exe" a inference_dist\fluid_install_dir.zip fluid_install_dir -r
goto :END

:FAILURE
echo BUILD FAILED
exit /b 1

:END
echo BUILD SUCCESSFULLY

ENDLOCAL

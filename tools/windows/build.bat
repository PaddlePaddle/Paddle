@ECHO OFF
SETLOCAL 
set source_path=%1
set PYTHON_DIR=%2
set WITH_GPU=%3
set WITH_MKL=%4
set WITH_AVX=%5
set BATDIR=%6
set release_dir=%7

set PADDLE_VERSION=1.3.0

REM setup msbuild env
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

REM set proxy
set http_proxy=%HTTP_PROXY%
set https_proxy=%HTTP_PROXY%

REM set python env
for /f "tokens=1,2,* delims=\\" %%a in ("%PYTHON_DIR%") do (
	set c1=%%a
	set c2=%%b
)
set PYTHONV=%c2%

if "%WITH_GPU%"=="ON" (
    set PLAT=GPU
) else (
    set PLAT=CPU
)

if "%WITH_MKL%"=="ON" (
    set BLAS=MKL
) else (
    set BLAS=OPEN
)

if "%WITH_AVX%"=="ON" (
    set INS=AVX
) else (
    set INS=NOAVX
)
if %errorlevel% NEQ 0 GOTO END

mkdir third_party\install
xcopy /s /e /y /c win\Release\third_party\ third_party\
if %errorlevel% NEQ 0 GOTO END

REM start the cmake process to generate the windows projects
cmake ..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DWITH_STATIC_LIB=ON    -DWITH_FLUID_ONLY=ON -DWITH_DSO=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON
if %errorlevel% NEQ 0 GOTO END

REM start the build process
msbuild /m /p:Configuration=Release third_party.vcxproj
if %errorlevel% NEQ 0 GOTO END
msbuild /m /p:Configuration=Release paddle.sln
if %errorlevel% NEQ 0 GOTO END

REM begin to verify
set "pub_path=%release_dir%\package_%PYTHONV%_%PLAT%_%BLAS%_%INS%"
mkdir %pub_path%
if %errorlevel% NEQ 0 GOTO END

set "tmpdir=%TMP%\verify_%PYTHONV%_%PLAT%_%BLAS%_%INS%"
echo tmpdir %tmpdir%
rmdir /s /q %tmpdir%
mkdir %tmpdir%
cd /d %tmpdir%
%PYTHON_DIR%\python.exe -m pip install --upgrade pip
%PYTHON_DIR%\python.exe -m pip install virtualenv
%PYTHON_DIR%\Scripts\virtualenv paddletest
call paddletest\Scripts\activate.bat
dir /s /b %dst_path%\python\dist\*.whl > whl_file.txt
set /p PADDLE_WHL_FILE_WIN=< whl_file.txt
if "%WITH_GPU%"=="ON" (
    pip uninstall -y paddlepaddle-gpu
) else (
    pip uninstall -y paddlepaddle
)
pip install %PADDLE_WHL_FILE_WIN%
echo import paddle.fluid;print(paddle.__version__) > test_whl.py
python test_whl.py
if %errorlevel% NEQ 0 GOTO END

REM copy %PADDLE_WHL_FILE_WIN% %pub_path%
copy /Y %PADDLE_WHL_FILE_WIN% %pub_path%
if %errorlevel% NEQ 0 GOTO END

:END
ENDLOCAL


@ECHO OFF
set /p source_path="Please input the dst path : "

if "%source_path%"=="" GOTO END

mkdir %source_path%
cd /d %source_path%
if %errorlevel% NEQ 0 GOTO END

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

set http_proxy=http://172.19.57.45:3128
set https_proxy=http://172.19.57.45:3128

echo "begin to download the source code from https://github.com/paddlepaddle/paddle"
git clone https://github.com/paddlepaddle/paddle
cd paddle
git checkout release/1.3
git pull
echo "download done!!!"

set /p PYTHON_DIR="Please input the python path(c:\Python27) : "
if "%PYTHON_DIR%"=="" (
    set PYTHON_DIR=c:\Python27
)
set /p WITH_GPU="Enable GPU (ON/OFF) : "
if "%WITH_MKL%"=="" (set WITH_MKL=ON)
set /p WITH_MKL="Enable MKL (ON/OFF) : "
if "%WITH_GPU%"=="" (set WITH_GPU=ON)
set /p WITH_AVX="Enable AVX (ON/OFF) : "
if "%WITH_AVX%"=="" (set WITH_AVX=ON)

echo "begin to do build..."
cd ..
set start_path=%~dp0
echo %start_path%

set "release_dir=%source_path%\paddle_release"
mkdir %release_dir%

REM source_path PYTHON_DIR WITH_GPU WITH_MKL WITH_AVX BATDIR
call %start_path%build.bat %source_path% c:\python27 ON ON ON %start_path% %release_dir%

:END


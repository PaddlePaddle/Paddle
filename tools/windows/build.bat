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

call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

REM set http_proxy=http://172.19.57.45:3128
REM set https_proxy=http://172.19.57.45:3128

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

set "dst_path=%source_path%\build_%PYTHONV%_%PLAT%_%BLAS%_%INS%"
echo %dst_path%
set "pub_path=%release_dir%\build_%PYTHONV%_%PLAT%_%BLAS%_%INS%"
echo %pub_path%
mkdir %pub_path%

mkdir %dst_path%
cd /d %dst_path%
if %errorlevel% NEQ 0 GOTO END

echo "begin to do build..."
"%BATDIR%\7z.exe" x "%BATDIR%\third_party.rar" -aoa -o.

cmake ..\Paddle -G "Visual Studio 14 2015 Win64" -DWITH_GPU=%WITH_GPU% -DWITH_MKL=%WITH_MKL% -DWITH_AVX=ON -DWITH_STATIC_LIB=ON    -DWITH_FLUID_ONLY=ON -DWITH_DSO=ON -DPYTHON_INCLUDE_DIR=%PYTHON_DIR%\include\ -DPYTHON_LIBRARY=%PYTHON_DIR%\libs\ -DPYTHON_EXECUTABLE=%PYTHON_DIR%\python.exe -DCMAKE_BUILD_TYPE=Release -DWITH_TESTING=OFF -DWITH_PYTHON=ON

msbuild /m /p:Configuration=Release third_party.vcxproj
msbuild /m /p:Configuration=Release paddle.sln
echo "build done!!!"

set "verifybat=%6\verify.bat"
call %verifybat% %1 %2 %3 %4 %5 %6 %7

:END
ENDLOCAL


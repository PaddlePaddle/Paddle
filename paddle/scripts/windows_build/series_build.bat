@ECHO OFF

set "source_path=%1"

if "%source_path%"=="" (
    set /p source_path="Please input the dst path : "
)

if not exist %source_path% mkdir %source_path%
cd /d %source_path%
if %errorlevel% NEQ 0 GOTO END
set PATH=%~dp0;%PATH%


set "release_dir=%source_path%\paddle_release"
mkdir %release_dir%

set http_proxy=http://172.19.57.45:3128
set https_proxy=http://172.19.57.45:3128
REM set http_proxy=http://172.19.56.199:3128
REM set https_proxy=http://172.19.56.199:3128
rem set http_proxy=http://182.61.163.38:31283128
rem set https_proxy=http://182.61.163.38:
set PADDLE_VERSION=2.0.0-alpha0

echo "begin to download the source code from https://github.com/paddlepaddle/paddle"
git clone https://github.com/paddlepaddle/paddle
cd paddle
git checkout paddle\fluid\framework\commit.h.in 
git checkout cmake\version.cmake
rem git checkout develop
git pull
REM git checkout release/1.5
git checkout v2.0.0-alpha0
rem git checkout develop
git pull
sed -i "s/@PADDLE_VERSION@/%PADDLE_VERSION%/g" paddle\fluid\framework\commit.h.in
sed -i "s/add_definitions(-DPADDLE_VERSION=\${PADDLE_VERSION})//g" cmake\version.cmake
echo "download done!!!"

cd ..
set start_path=%~dp0
echo %start_path%

REM source_path PYTHON_DIR WITH_GPU WITH_MKL WITH_AVX BATDIR


set DEFAULT_CUDA=d:/v9.0

rem call %start_path%build.bat %source_path% c:\python36 OFF ON %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
rem for /D %%i in (d:/v8.0 d:/v9.0 d:/v9.2 d:/v10.0) do (
REM for /D %%i in (d:/v10.0 d:/v9.0 d:/v8.0) do (
REM for /D %%i in (d:/v10.0 d:/v9.0) do (
for /D %%i in ( d:/v10.0 ) do (
call %start_path%build.bat %source_path% c:\python27 ON ON OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
call %start_path%build.bat %source_path% c:\python35 ON ON OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python36 ON ON OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python37 ON ON OFF %PADDLE_VERSION% %start_path% %release_dir% %%i


rem call %start_path%build.bat %source_path% c:\python27 ON OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python35 ON OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python36 ON OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python37 ON OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %%i

rem call %start_path%build.bat %source_path% c:\python37 ON ON ON %PADDLE_VERSION% %start_path% %release_dir% %%i
rem call %start_path%build.bat %source_path% c:\python37 ON OFF ON %PADDLE_VERSION% %start_path% %release_dir% %%i
)
rem build cpu first


rem call %start_path%build.bat %source_path% c:\python27 OFF ON OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
rem call %start_path%build.bat %source_path% c:\python27 OFF OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%

rem call %start_path%build.bat %source_path% c:\python35 OFF ON OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
rem call %start_path%build.bat %source_path% c:\python35 OFF OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%

rem call %start_path%build.bat %source_path% c:\python36 OFF ON OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
REM call %start_path%build.bat %source_path% c:\python36 OFF OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%

rem call %start_path%build.bat %source_path% c:\python37 OFF ON OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
REM call %start_path%build.bat %source_path% c:\python37 OFF OFF OFF %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%

rem call %start_path%build.bat %source_path% c:\python37 OFF ON ON %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
REM call %start_path%build.bat %source_path% c:\python37 OFF OFF ON %PADDLE_VERSION% %start_path% %release_dir% %DEFAULT_CUDA%
:END


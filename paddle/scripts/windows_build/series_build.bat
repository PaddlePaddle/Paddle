@ECHO OFF

for /f "eol=# delims== tokens=1,2" %%i in (config.ini) do (
    set %%i=%%j
)

set "source_path=%1"

if "%source_path%"=="" (
    set /p source_path="Please input the dst path :  =======>"
)

if not exist %source_path% mkdir %source_path%
cd /d %source_path%
if %errorlevel% NEQ 0 GOTO END


echo "begin to download the source code from https://github.com/paddlepaddle/paddle"
git clone https://github.com/PaddlePaddle/Paddle
cd Paddle
git checkout %BRANCH%
rem sed -i "s/@PADDLE_VERSION@/%PADDLE_VERSION%/g" paddle\fluid\framework\commit.h.in
rem sed -i "s/add_definitions(-DPADDLE_VERSION=\${PADDLE_VERSION})//g" cmake\version.cmake
echo "download done!!!"

cd ..
set start_path=%~dp0
echo %start_path%


echo Init Visual Studio Env
call %vcvarsall_dir% amd64

rem source_path PYTHON_DIR WITH_GPU WITH_MKL ON_INFER PADDLE_VERSION BATDIR CUDA_DIR


rem ==============build GPU============
for /D %%i in ( %CUDA_PATH% ) do (
    for /D %%j in ( %PYTHON_PATH% ) do (
        call %start_path%build.bat %source_path% %%j ON ON OFF %PADDLE_VERSION% %start_path% %%i
        call %start_path%build.bat %source_path% %%j ON OFF OFF %PADDLE_VERSION% %start_path% %%i
    )
    rem ===build inference library===
    call %start_path%build.bat %source_path% %PYTHON3_PATH% ON ON ON %PADDLE_VERSION% %start_path% %release_dir% %%i
    call %start_path%build.bat %source_path% %PYTHON3_PATH% ON OFF ON %PADDLE_VERSION% %start_path% %release_dir% %%i
)


rem ==============build CPU=============

for /D %%j in ( %PYTHON_PATH% ) do (
    call %start_path%build.bat %source_path% %%j OFF ON OFF %PADDLE_VERSION% %start_path% NEEDLESS
    call %start_path%build.bat %source_path% %%j OFF OFF OFF %PADDLE_VERSION% %start_path% NEEDLESS
)
rem ===build inference library===
call %start_path%build.bat %source_path% %PYTHON3_PATH% OFF ON ON %PADDLE_VERSION% %start_path% %release_dir% NEEDLESS
call %start_path%build.bat %source_path% %PYTHON3_PATH% OFF OFF ON %PADDLE_VERSION% %start_path% %release_dir% NEEDLESS

:END
rem reset environment variable
for /f "eol=# delims== tokens=1,2" %%i in (config.ini) do (
    set %%i=
)
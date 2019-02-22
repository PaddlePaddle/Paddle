@ECHO OFF

set "source_path=%1"

if "%source_path%"=="" (
    set /p source_path="Please input the dst path : "
)

mkdir %source_path%
cd /d %source_path%
if %errorlevel% NEQ 0 GOTO END

set "release_dir=%source_path%\paddle_release"
mkdir %release_dir%

REM set http_proxy=http://172.19.57.45:3128
REM set https_proxy=http://172.19.57.45:3128

echo "begin to download the source code from https://github.com/paddlepaddle/paddle"
git clone https://github.com/paddlepaddle/paddle
cd paddle
REM git checkout CMakeLists.txt
git checkout paddle\fluid\framework\ir\graph.h
rem git checkout release/1.3
git checkout V1.3.0
git pull
REM echo set(CMAKE_SUPPRESS_REGENERATION true) >> CMakelists.txt
sed -i "s/ir::Node::kControlDepVarName, node_set_.size());/static_cast<const char *>(ir::Node::kControlDepVarName),node_set_.size());/g" paddle\fluid\framework\ir\graph.h
echo "download done!!!"

cd ..
set start_path=%~dp0
echo %start_path%

REM source_path PYTHON_DIR WITH_GPU WITH_MKL WITH_AVX BATDIR
if 0==1 (
echo nothnig
)
call %start_path%build.bat %source_path% c:\python27 ON ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 ON OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 ON ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 ON OFF OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 OFF ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 OFF OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 OFF ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python27 OFF OFF OFF %start_path% %release_dir%

call %start_path%build.bat %source_path% c:\python35 ON ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 ON OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 ON ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 ON OFF OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 OFF ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 OFF OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 OFF ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python35 OFF OFF OFF %start_path% %release_dir%

call %start_path%build.bat %source_path% c:\python36 ON ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 ON OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 ON ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 ON OFF OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 OFF ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 OFF OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 OFF ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python36 OFF OFF OFF %start_path% %release_dir%

call %start_path%build.bat %source_path% c:\python37 ON ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 ON OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 ON ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 ON OFF OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 OFF ON ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 OFF OFF ON %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 OFF ON OFF %start_path% %release_dir%
call %start_path%build.bat %source_path% c:\python37 OFF OFF OFF %start_path% %release_dir%

:END


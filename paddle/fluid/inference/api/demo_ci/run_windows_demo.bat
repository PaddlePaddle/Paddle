@echo off
setlocal
set source_path=%~dp0
set build_path=%~dp0\build

setlocal enabledelayedexpansion

rem set gpu_inference
SET /P gpu_inference="Use GPU_inference_lib or not(Y/N), default: N   =======>"
IF /i "%gpu_inference%"=="y" (
  SET gpu_inference=Y
) else (
  SET gpu_inference=N
)

SET /P use_mkl="Use MKL or not (Y/N), default: Y   =======>"
if /i "%use_mkl%"=="N" (
  set use_mkl=N
) else (
  set use_mkl=Y
)

:set_paddle_infernece_lib
SET /P paddle_infernece_lib="Please input the path of paddle inference library, such as D:\paddle_inference_install_dir   =======>"
set tmp_var=!paddle_infernece_lib!
call:remove_space
set paddle_infernece_lib=!tmp_var!
IF NOT EXIST "%paddle_infernece_lib%" (
echo "------------%paddle_infernece_lib% not exist------------"
goto set_paddle_infernece_lib
)

IF "%use_mkl%"=="N" (
  IF NOT EXIST "%paddle_infernece_lib%\third_party\install\openblas" (
    echo "------------It's not a OpenBlas inference library------------"
    goto:eof
  )
) else (
  IF NOT EXIST "%paddle_infernece_lib%\third_party\install\mklml" (
    echo "------------It's not a MKL inference library------------"
    goto:eof
  )
)

:set_path_cuda
if /i "!gpu_inference!"=="Y" (
    SET /P cuda_lib_dir="Please input the path of cuda libraries, such as D:\cuda\lib\x64   =======>"
    set tmp_var=!cuda_lib_dir!
    call:remove_space
    set cuda_lib_dir=!tmp_var!
    IF NOT EXIST "!cuda_lib_dir!" (
        echo "------------!cuda_lib_dir!not exist------------"
        goto set_path_cuda
    )
)

rem set_use_gpu
if /i "!gpu_inference!"=="Y" (
    SET /P use_gpu="Use GPU or not(Y/N), default: N   =======>"
)

if /i "%use_gpu%"=="Y" (
  set use_gpu=Y
) else (
  set use_gpu=N
)

rem set_path_vs_command_prompt
:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat   =======>"
set tmp_var=!vcvarsall_dir!
call:remove_space
set vcvarsall_dir=!tmp_var!
IF NOT EXIST "%vcvarsall_dir%" (
    echo "------------%vcvarsall_dir% not exist------------"
    goto set_vcvarsall_dir
)

rem set_demo_name
:set_demo_name
SET /P demo_name="Please input the demo name, default: windows_mobilenet  =======>"
if   "%demo_name%"==""  set demo_name=windows_mobilenet
IF NOT EXIST "%source_path%\%demo_name%.cc" (
    echo "------------%source_path%\%demo_name%.cc not exist------------"
    goto set_demo_name
)
if "%demo_name%"=="windows_mobilenet" set model_name=mobilenet
if "%demo_name%"=="vis_demo" set model_name=mobilenet
if "%demo_name%"=="simple_on_word2vec" set model_name=word2vec.inference.model
if "%demo_name%"=="trt_mobilenet_demo" set model_name=mobilenet

rem download model
if NOT EXIST "%source_path%\%model_name%.tar.gz" (
  if "%model_name%"=="mobilenet" (
     call:download_model_mobilenet
  )
  if "%model_name%"=="word2vec.inference.model" (
     call:download_model_word2vec
  )
)

if EXIST "%source_path%\%model_name%.tar.gz" (
  if NOT EXIST "%source_path%\%model_name%" (
    SET /P python_path="Please input the path of python.exe, such as C:\Python37\python.exe =======>"
    set tmp_var=!python_path!
    call:remove_space
    set python_path=!tmp_var!
    if "!python_path!"=="" (
      set python_path=python.exe
    ) else (
      if NOT exist "!python_path!" (
        echo "------------!python_path! not exist------------"
        goto:eof
      )
    )
    md %source_path%\%model_name%
    !python_path! %source_path%\untar_model.py %source_path%\%model_name%.tar.gz %source_path%\%model_name%

    SET error_code=N
    if "%model_name%"=="mobilenet" (
      if NOT EXIST "%source_path%\%model_name%\model" set error_code=Y
    ) else (
      if NOT EXIST "%source_path%\%model_name%\%model_name%" set error_code=Y
    )
    if  "!error_code!"=="Y"  (
       echo "========= Unzip %model_name%.tar.gz failed ======="
       del /f /s /q "%source_path%\%model_name%\*.*" >nul 2>&1
       rd /s /q  "%source_path%\%model_name%" >nul 2>&1
       goto:eof
    )
  )
)

echo "=================================================================="
echo.
echo "use_gpu_inference=%gpu_inference%"
echo.
echo "use_mkl=%use_mkl%"
echo.
echo "use_gpu=%use_gpu%"
echo.
echo "paddle_infernece_lib=%paddle_infernece_lib%"
echo.
IF /i "%gpu_inference%"=="y" (
  echo "cuda_lib_dir=%cuda_lib_dir%"
  echo.
)
echo "vs_vcvarsall_dir=%vcvarsall_dir%"
echo.
echo "demo_name=%demo_name%"
echo.
if NOT "!python_path!"=="" (
  echo "python_path=!python_path!"
  echo.
)
echo "===================================================================="
pause


rem compile and run demo

if NOT EXIST "%build_path%" (
    md %build_path%
    cd %build_path%
) else (
    del /f /s /q "%build_path%\*.*" >nul 2>&1
    rd /s /q  "%build_path%" >nul 2>&1
    md %build_path%
    cd %build_path%
)

if /i "%use_mkl%"=="N" (
  set use_mkl=OFF
) else (
  set use_mkl=ON
)

if /i "%gpu_inference%"=="Y" (
    if  "%demo_name%"=="trt_mobilenet_demo" (
      cmake .. -G "Visual Studio 15 2017 Win64"  -T host=x64 -DWITH_GPU=ON ^
      -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
      -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON -DCUDA_LIB="%cuda_lib_dir%" -DUSE_TENSORRT=ON
    ) else (
      cmake .. -G "Visual Studio 15 2017 Win64"  -T host=x64 -DWITH_GPU=ON ^
      -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
      -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON -DCUDA_LIB="%cuda_lib_dir%"
    )
) else (
    cmake .. -G "Visual Studio 15 2017 Win64"  -T host=x64 -DWITH_GPU=OFF ^
    -DWITH_MKL=%use_mkl% -DWITH_STATIC_LIB=ON -DCMAKE_BUILD_TYPE=Release -DDEMO_NAME=%demo_name% ^
    -DPADDLE_LIB="%paddle_infernece_lib%" -DMSVC_STATIC_CRT=ON
)

call "%vcvarsall_dir%" amd64
msbuild /m /p:Configuration=Release %demo_name%.vcxproj

if /i "%use_gpu%"=="Y" (
  SET use_gpu=true
) else (
  SET use_gpu=false
)

if exist "%build_path%\Release\%demo_name%.exe" (
  cd %build_path%\Release
  set GLOG_v=4
  if "%demo_name%"=="simple_on_word2vec" (
      %demo_name%.exe --dirname="%source_path%\%model_name%\%model_name%" --use_gpu="%use_gpu%"
  ) else (
    if "%demo_name%"=="windows_mobilenet" (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --use_gpu="%use_gpu%"
    ) else (
      if "%demo_name%"=="trt_mobilenet_demo" (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --data=%source_path%\%model_name%\data.txt ^
        --refer=%source_path%\%model_name%\result.txt
      ) else (
        %demo_name%.exe --modeldir="%source_path%\%model_name%\model" --data=%source_path%\%model_name%\data.txt ^
        --refer=%source_path%\%model_name%\result.txt --use_gpu="%use_gpu%"
      )
    )
  )
) else (
  echo "=========compilation fails!!=========="
)
echo.&pause&goto:eof

:download_model_mobilenet
powershell.exe (new-object System.Net.WebClient).DownloadFile('http://paddlemodels.bj.bcebos.com//inference-vis-demos/mobilenet.tar.gz', ^
'%source_path%\mobilenet.tar.gz')
goto:eof

:download_model_word2vec
powershell.exe (new-object System.Net.WebClient).DownloadFile('http://paddle-inference-dist.bj.bcebos.com/word2vec.inference.model.tar.gz', ^
'%source_path%\word2vec.inference.model.tar.gz')
goto:eof

:remove_space
:remove_left_space
if "%tmp_var:~0,1%"==" " (
    set "tmp_var=%tmp_var:~1%"
    goto remove_left_space
)

:remove_right_space
if "%tmp_var:~-1%"==" " (
    set "tmp_var=%tmp_var:~0,-1%"
    goto remove_left_space
)
goto:eof

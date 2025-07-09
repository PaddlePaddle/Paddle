setlocal enabledelayedexpansion

set work_dir=%cd%
echo work_dir=%work_dir%>> %GITHUB_ENV%
if not defined cache_dir (
    set "cache_dir=%work_dir%\..\cache"
    echo cache_dir=%cache_dir%>> %GITHUB_ENV%
)

if not exist %cache_dir% mkdir %cache_dir%
if not exist %cache_dir%\tools (
    cd /d %cache_dir%
    python -m pip install wget
    python -c "import wget;wget.download('https://paddle-ci.gz.bcebos.com/window_requirement/tools.zip')"
    tar xf tools.zip
    cd /d %work_dir%
)

pip config set global.trusted-host pypi.org
pip config set global.trusted-host files.pythonhosted.org
pip config set global.trusted-host pypi.python.org
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
git config --global core.longpaths true
git config --global user.name "PaddleCI"
git config --global user.email "paddle_ci@example.com"

git remote add upstream https://github.com/PaddlePaddle/Paddle.git

git --no-pager pull upstream %BRANCH% --no-edit
if %errorlevel% NEQ 0 exit /b 1
if exist .git\index.lock del .git\index.lock 2>NUL
if not defined GENERATOR echo GENERATOR="Visual Studio 15 2017 Win64">> %GITHUB_ENV%
if not defined WITH_TENSORRT echo WITH_TENSORRT=ON>> %GITHUB_ENV%
if not defined TENSORRT_ROOT echo TENSORRT_ROOT=D:/TensorRT>> %GITHUB_ENV%
if not defined WITH_GPU echo WITH_GPU=ON>> %GITHUB_ENV%
if not defined WITH_MKL echo WITH_MKL=ON>> %GITHUB_ENV%
if not defined WITH_AVX echo WITH_AVX=ON>> %GITHUB_ENV%
if not defined WITH_TESTING echo WITH_TESTING=ON>> %GITHUB_ENV%
if not defined MSVC_STATIC_CRT echo MSVC_STATIC_CRT=ON>> %GITHUB_ENV%
if not defined WITH_PYTHON (
    set WITH_PYTHON=ON
    echo WITH_PYTHON=ON>> %GITHUB_ENV%
)
if not defined ON_INFER echo ON_INFER=ON>> %GITHUB_ENV%
if not defined WITH_ONNXRUNTIME echo WITH_ONNXRUNTIME=OFF>> %GITHUB_ENV%
if not defined WITH_INFERENCE_API_TEST echo WITH_INFERENCE_API_TEST=ON>> %GITHUB_ENV%
if not defined WITH_STATIC_LIB echo WITH_STATIC_LIB=ON>> %GITHUB_ENV%
if not defined WITH_UNITY_BUILD echo WITH_UNITY_BUILD=OFF>> %GITHUB_ENV%
if not defined NEW_RELEASE_ALL echo NEW_RELEASE_ALL=ON>> %GITHUB_ENV%
if not defined NEW_RELEASE_PYPI echo NEW_RELEASE_PYPI=OFF>> %GITHUB_ENV%
if not defined NEW_RELEASE_JIT echo NEW_RELEASE_JIT=OFF>> %GITHUB_ENV%
if not defined WITH_CPP_TEST echo WITH_CPP_TEST=ON>> %GITHUB_ENV%
if not defined WITH_NIGHTLY_BUILD echo WITH_NIGHTLY_BUILD=OFF>> %GITHUB_ENV%

if not defined WITH_TPCACHE echo WITH_TPCACHE=OFF>> %GITHUB_ENV%
if not defined WITH_CACHE echo WITH_CACHE=OFF>> %GITHUB_ENV%
if not defined WITH_SCCACHE echo WITH_SCCACHE=OFF>> %GITHUB_ENV%
if not defined INFERENCE_DEMO_INSTALL_DIR echo INFERENCE_DEMO_INSTALL_DIR=%cache_dir:\=/%/inference_demo>> %GITHUB_ENV%
if not defined LOG_LEVEL echo LOG_LEVEL=normal>> %GITHUB_ENV%
if not defined PRECISION_TEST echo PRECISION_TEST=OFF>> %GITHUB_ENV%
if not defined WIN_UNITTEST_LEVEL echo WIN_UNITTEST_LEVEL=2>> %GITHUB_ENV%
rem LEVEL 0: For unittests unrelated to CUDA/TRT or unittests without GPU memory, only run on
rem Windows-Infernece(CUDA 11.2), skip them on Windows-GPU(CUDA 12.0)
rem LEVEL 1: For unittests unrelated to CUDA/TRT, only run on Windows-Infernece(CUDA 11.2),
rem skip them on Windows-GPU(CUDA 12.0)
rem LEVEL 2: run all test
if not defined NIGHTLY_MODE echo NIGHTLY_MODE=OFF>> %GITHUB_ENV%
if not defined PYTHON_ROOT echo PYTHON_ROOT=C:\Python38>> %GITHUB_ENV%
if not defined BUILD_DIR echo BUILD_DIR=build>> %GITHUB_ENV%
if not defined TEST_INFERENCE echo TEST_INFERENCE=ON>> %GITHUB_ENV%
if not defined WITH_PIP_CUDA_LIBRARIES echo WITH_PIP_CUDA_LIBRARIES=OFF>> %GITHUB_ENV%

echo UPLOAD_TP_FILE=OFF>> %GITHUB_ENV%
echo UPLOAD_TP_CODE=OFF>> %GITHUB_ENV%

echo error_code=0 >> %GITHUB_ENV%
type %cache_dir%\error_code.txt

rem ------initialize set git config------
git config --global core.longpaths true

rem ------initialize the python environment------
set "PYTHON_VENV_ROOT=%cache_dir%\python_venv"
echo PYTHON_VENV_ROOT=%PYTHON_VENV_ROOT%>> %GITHUB_ENV%
if not exist %PYTHON_VENV_ROOT% mkdir %PYTHON_VENV_ROOT%
set "PYTHON_EXECUTABLE=%PYTHON_VENV_ROOT%\Scripts\python.exe"
echo PYTHON_EXECUTABLE=%PYTHON_EXECUTABLE%>> %GITHUB_ENV%
%PYTHON_ROOT%\python.exe -m venv --clear %PYTHON_VENV_ROOT%
call "%PYTHON_VENV_ROOT%\Scripts\activate.bat"
if %ERRORLEVEL% NEQ 0 (
    echo activate python virtual environment failed!
    exit /b 5
)
python -m pip install wget
if "%WITH_PYTHON%" == "ON" (
    where python
    where pip
    python -m pip install --upgrade pip
    python -m pip install -r %work_dir%\paddle\scripts\compile_requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo pip install compile_requirements.txt failed!
        exit /b 5
    )
    python -m pip install -r %work_dir%\python\requirements.txt
    if !ERRORLEVEL! NEQ 0 (
        echo pip install requirements.txt failed!
        exit /b 5
    )
)

:: Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::
:: ===============================
:: Build Paddle compile environment
:: ===============================
:: Description:
::
::   Install compile environment for xly CI.
::
::   Include:
::     1. CMake 3.18.0
::     2. Git 2.28.0
::     3. Python 3.8.3\3.9.7\3.10.0
::     4. Visual Studio 2017 Community
::     5. CUDA 11.2
::     6. java jre
::     7. sccache
::     8. TensorRT
::     9. xly agent

:: Echo command is not required.
@echo off
cd /d %~dp0%

SET INSTALL_FLAG=%1
::  0. PR_CI_Windows_OPENBLAS
::  1. PR_CI_Windows_Inference，will install VS2019, CUDA11.2
::  2. PR_CI_Windows，CUDA10.2


:: ===== start step 0: wget tool =====
:: Download wget for windows when there is not wget tool.
echo ">>>>>>>> step [0/9]: wget tool"
wget --help > nul 2> nul || call:install_wget
goto cmake

:install_wget
echo There is not wget in this PC, will download wget 1.21.4.
echo Download package from https://eternallybored.org/misc/wget/1.21.4/64/wget.exe ...
certutil -urlcache -split -f https://eternallybored.org/misc/wget/1.21.4/64/wget.exe > nul 2> nul
if %errorlevel% == 0 (
  echo Download wget tool into %cd% success.
) else (
  echo Error***** Download wget tool failed, please download it before rerun.
  exit /b 1
)
goto :eof
:: ===== end step 0: wget tool =====

:: ===== start step 1: cmake =====
:: Download CMake-3.18.0 and add in PATH when it not installed.
:: TODO: limit version >= 3.18.0
:cmake
echo ">>>>>>>> step [1/9]: CMake 3.18.0"
cmake --help > nul 2> nul || call :install_cmake
goto git

:install_cmake
echo There is not cmake in this PC, will install cmake-3.18.0.
echo Download package from https://cmake.org/files/v3.18/cmake-3.18.0-win64-x64.msi ...
wget -O cmake-3.18.0-win64-x64.msi https://cmake.org/files/v3.18/cmake-3.18.0-win64-x64.msi
echo Install cmake-3.18.0 ...
:: /passive [silent installation]
:: /norestart [do not restart]
:: ADD_CMAKE_TO_PATH = System [add CMake to the system PATH for all users]
start /wait cmake-3.18.0-win64-x64.msi /passive /norestart ADD_CMAKE_TO_PATH=System
if %errorlevel% == 0 (
  echo Install CMake-3.18.0 success!
) else (
  echo Error***** Install Cmake-3.18.0 failed, please re-install it manually.
)
goto :eof
:: ===== end step 1: cmake =====

:: ===== start step 2: Git =====
:: Download Git-2.28.0 and add in PATH when it not installed.
:: TODO: limit version >= 2.28.0
:git
echo ">>>>>>>> step [2/9]: Git 2.28.0"
git --help > nul 2> nul || call :install_git
goto python

:install_git
echo There is not git in this PC, will install Git-2.28.0.
echo Download package from https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe ...
wget -O Git-2.28.0-64-bit.exe https://github.com/git-for-windows/git/releases/download/v2.28.0.windows.1/Git-2.28.0-64-bit.exe
echo Install Git-2.28.0 ...
:: /SILENT [silent install]
:: /ALLUSERS [add path for all users]
:: /NORESTART [do not restart]
start /wait Git-2.28.0-64-bit.exe /SILENT /ALLUSERS /NORESTART
if %errorlevel% == 0 (
  echo Install Git-2.28.0 success!
) else (
  echo Error***** Install Git-2.28.0 failed, please re-install it manually.
)
goto :eof
:: ===== end step 2: Git =====

:: ===== start step 3: Python =====
:: Download Python and add in PATH when it not installed.
:: TODO: limit version >= 3.8.3
:python
echo ">>>>>>>> step [3/9]: Python"
where python 2>&1 | findstr /C:"Python38" > nul 2> nul || call :install_python3.8.3
where python 2>&1 | findstr /C:"Python39" > nul 2> nul || call :install_python3.9.7
where python 2>&1 | findstr /C:"Python310" > nul 2> nul || call :install_python3.10.0
goto vs

:install_python3.8.3
echo There is not Python in this PC, will install Python-3.8.3
echo Download package from https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe ...
wget -O python-3.8.3-amd64.exe https://www.python.org/ftp/python/3.8.3/python-3.8.3-amd64.exe
echo Install Python-3.8.3 ...
:: /passive [silent install]
:: InstallAllUsers [add path for all users]
:: PrependPath [add script/install into PATH]
:: TargetDir [install directory]
start /wait python-3.8.3-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python38
if %errorlevel% == 0 (
  echo Install python-3.8.3 success!
) else (
  echo Error***** Install python-3.8.3 failed, please re-install it manually.
)
goto :eof

:install_python3.9.7
echo There is not Python in this PC, will install Python-3.9.7
echo Download package from https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe ...
wget -O python-3.9.7-amd64.exe https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe
echo Install Python-3.9.7 ...
start /wait python-3.9.7-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python39
if %errorlevel% == 0 (
  echo Install python-3.9.7 success!
) else (
  echo Error***** Install python-3.9.7 failed, please re-install it manually.
)
goto :eof

:install_python3.10.0
echo There is not Python in this PC, will install Python-3.10.0
echo Download package from https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe ...
wget -O python-3.10.0-amd64.exe https://www.python.org/ftp/python/3.10.0/python-3.10.0-amd64.exe
echo Install Python-3.10.0 ...
start /wait python-3.10.0-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python310
if %errorlevel% == 0 (
  echo Install python-3.10.0 success!
) else (
  echo Error***** Install python-3.10.0 failed, please re-install it manually.
)
goto :eof
:: ===== end step 3: Python =====

:: ===== start step 4: Visual Studio Community =====
:: Download Visual Studio 2017 when it not installed.
:vs
echo ">>>>>>>> step [4/9]: Visual Studio"
if %INSTALL_FLAG% == 1 (
  cmd /C "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"  > nul 2> nul || call :install_visual_studio2019
) else (
  cmd /C "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"  > nul 2> nul || call :install_visual_studio2017
)
goto :cuda

:install_visual_studio2017
echo There is not Visual Studio in this PC, will install VS2017.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/VS2017/vs_Community.exe"
wget -O vs_Community.exe "https://paddle-ci.gz.bcebos.com/window_requirement/VS2017/vs_Community.exe"
echo Install Visual Studio 2017 ...
:: /passive [silent install]
:: /norestart [no restart]
:: /NoRefresh [no refresh]
:: /InstallSelectableItems NativeLanguageSupport_Group [select Visual C++ for installing]
start /wait vs_Community.exe --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.Universal --includeRecommended
if %errorlevel% == 0 (
  echo Install Visual Studio 2017 success!
) else (
  echo Error***** Install Visual Studio 2017 failed, please re-install it manually.
)
goto :eof

:install_visual_studio2019
echo There is not Visual Studio in this PC, will install VS2019.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/vs_community_2019.exe"
wget -O vs_community_2019.exe "https://paddle-ci.gz.bcebos.com/window_requirement/vs_community_2019.exe"
echo Install Visual Studio 2019 ...
start /wait vs_community_2019.exe --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.Universal --includeRecommended
if %errorlevel% == 0 (
  echo Install Visual Studio 2019 success!
) else (
  echo Error***** Install Visual Studio 2019 failed, please re-install it manually.
)
goto :eof
:: ===== end step 4: Visual Studio Community =====

:: ===== start step 5: CUDA =====
:cuda
echo ">>>>>>>> step [5/9]: CUDA "
if %INSTALL_FLAG% == 1 (
    cmd /C nvcc --version 2> nul | findstr /C:"11.2" > nul 2> nul || call :install_cuda112
)
if %INSTALL_FLAG% == 2 (
    cmd /C nvcc --version 2> nul | findstr /C:"10.2" > nul 2> nul || call :install_cuda102
)
goto java-jre

:install_cuda102
echo There is not CUDA in this PC, will install CUDA-10.2.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_10.2.89_441.22_win10.exe"
wget -O cuda_installer.exe "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_10.2.89_441.22_win10.exe"
echo Install CUDA-10.2 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-10.2 success!
) else (
  echo Error***** Install CUDA-10.2 failed, please re-install it manually.
  goto :eof
)
del cuda_installer.exe

echo Download cudnn from "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-10.2-windows10-x64-v7.6.5.32.zip"
wget -O cudnn-10.2-windows10-x64-v7.6.5.32.zip "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-10.2-windows10-x64-v7.6.5.32.zip"
tar xf cudnn-10.2-windows10-x64-v7.6.5.32.zip
xcopy /E /Y /R "cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin"
xcopy /E /Y /R "cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include"
xcopy /E /Y /R "cuda\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib"
rd /s /q cuda
goto :eof

:install_cuda112
echo There is not CUDA in this PC, will install CUDA-11.2.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_11.2.0_460.89_win10.exe"
wget -O cuda_installer.exe "https://paddle-ci.gz.bcebos.com/window_requirement/cuda_11.2.0_460.89_win10.exe"
echo Install CUDA-11.2 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-11.2 success!
) else (
  echo Error***** Install CUDA-11.2 failed, please re-install it manually.
  goto :eof
)
del cuda_installer.exe
echo Download cudnn from "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-11.2-windows-x64-v8.1.0.77.zip"
wget -O cudnn-11.2-windows-x64-v8.1.0.77.zip "https://paddle-ci.gz.bcebos.com/window_requirement/cudnn-11.2-windows-x64-v8.1.0.77.zip"
tar xf cudnn-11.2-windows-x64-v8.1.0.77.zip
xcopy /E /Y /R "cuda\bin\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"
xcopy /E /Y /R "cuda\include\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include"
xcopy /E /Y /R "cuda\lib\*" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib"
rd /s /q cuda
goto :eof
:: ===== end step 5: CUDA =====

:: ===== start step 6: java jre =====
:java-jre
echo ">>>>>>>> step [6/9]: java jre"
cmd /C java -version > nul 2> nul || call :install_java
goto sccache

:install_java
echo There is not java-jre in this PC, will install java-jre.
echo Download package from "https://paddle-ci.gz.bcebos.com/window_requirement/jre-8u261-windows-i586.exe"
wget -O jre-8u261-windows-x64.exe "https://paddle-ci.gz.bcebos.com/window_requirement/jre-8u261-windows-i586.exe"
echo Install java-jre ...
:: -s [silent install]
start /wait jre-8u261-windows-x64.exe /s
if %errorlevel% == 0 (
  echo Install java success!
) else (
  echo Error***** Install java failed, please re-install it manually.
)
del jre-8u261-windows-x64.exe
goto :eof
:: ===== end step 6: java jre =====

:: ===== start step 7: sccache on windowss =====
:sccache
echo ">>>>>>>> step [7/9]: sccache"
cmd /C sccache -V > nul 2> nul || call :download_sccache
goto tensorrt

:download_sccache
echo There is not sccache in this PC, will install sccache.
echo Download package from https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe
wget -O sccache.exe "https://paddle-ci.gz.bcebos.com/window_requirement/sccache.exe"
copy sccache.exe C:\Python38 /Y
goto :eof
:: ===== end step 7: sccache on windows =====

:: ===== start step 8: TensorRT =====
:tensorrt
echo ">>>>>>>> step [8/9]: TensorRT"
if %INSTALL_FLAG% == 2 (
  call :download_TensorRT_7_0_0_11
)
if %INSTALL_FLAG% == 1 (
  call :download_TensorRT_8_0_1_6
)
goto xly-agent

:download_TensorRT_7_0_0_11
if not exist TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6.zip wget -O TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6.zip ^
"https://paddle-ci.gz.bcebos.com/window_requirement/TensorRT-7.0.0.11.Windows10.x86_64.cuda-10.2.cudnn7.6.zip"
goto :eof

:download_TensorRT_8_0_1_6
if not exist TTensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip wget -O TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip ^
"https://paddle-ci.gz.bcebos.com/window_requirement/TensorRT-8.0.1.6.Windows10.x86_64.cuda-11.3.cudnn8.2.zip"
goto :eof
:: ===== end step 8: TensorRT =====

:: ===== start step 9: xly agent =====
:xly-agent
echo ">>>>>>>> step [9/9]: xly agent"
if not exist agent.jar wget -O agent.jar "https://xly.bce.baidu.com/sa_server/agent/v1/download?version=1.2.8"
:: ===== end step 9: xly agent =====

pause

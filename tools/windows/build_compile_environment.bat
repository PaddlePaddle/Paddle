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
:: Build Paddle compile enviroment
:: ===============================
:: Description:
::   
::   Install compile enviroment for xly CI.
::
::   Include:
::     1. CMake 3.17.0
::     2. Git 2.28.0
::     3. Python 3.7.8
::     4. Visual Studio 2015 with update 3
::     5. CUDA 10 [miss cudnn]
::     6. java jre [not complete]
::     7. xly agent [not complete]

:: Echo command is not required.
@echo off

:: ===== start step 0: wget tool =====
:: Download wget for windows when there is not wget tool.
echo ">>>>>>>> step [0/7]: wget tool"
wget --help > nul 2> nul || call:install_wget
goto cmake

:install_wget
echo There is not wget in this PC, will download wget 1.20.
echo Download package from https://eternallybored.org/misc/wget/1.20/64/wget.exe ...
certutil -urlcache -split -f https://eternallybored.org/misc/wget/1.20/64/wget.exe > nul 2> nul
if %errorlevel% == 0 (
  echo Download wget tool into %cd% success.
) else (
  echo Error***** Download wget tool failed, please download it before rerun.
  exit /b 1
) 
goto :eof
:: ===== end step 0: wget tool =====

:: ===== start step 1: cmake =====
:: Download CMake-3.17.0 and add in PATH when it not installed.
:: TODO: limit version >= 3.17.0
:cmake
echo ">>>>>>>> step [1/7]: CMake 3.17.0"
cmake --help > nul 2> nul || call :install_cmake
goto git

:install_cmake
echo There is not cmake in this PC, will install cmake-3.17.0.
echo Download package from https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi ...
wget -O cmake-3.17.0-win64-x64.msi https://cmake.org/files/v3.17/cmake-3.17.0-win64-x64.msi
echo Install cmake-3.17.0 ...
:: /passive [silent installation]
:: /norestart [do not restart]
:: ADD_CMAKE_TO_PATH = System [add CMake to the system PATH for all users]
start /wait cmake-3.17.0-win64-x64.msi /passive /norestart ADD_CMAKE_TO_PATH=System
if %errorlevel% == 0 (
  echo Install CMake-3.17.0 success!
) else (
  echo Error***** Install Cmake-3.17.0 failed, please re-install it manually.
)
del cmake-3.17.0-win64-x64.msi
goto :eof
:: ===== end step 1: cmake =====

:: ===== start step 2: Git =====
:: Download Git-2.28.0 and add in PATH when it not installed.
:: TODO: limit version >= 2.28.0
:git
echo ">>>>>>>> step [2/8]: Git 2.28.0"
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
del Git-2.28.0-64-bit.exe
goto :eof
:: ===== end step 2: Git =====

:: ===== start step 3: Python =====
:: Download Python-3.7.8 and add in PATH when it not installed.
:: TODO: limit version >= 3.7.8
:python
echo ">>>>>>>> step [3/7]: Python 3.7.8"
python -V 2>&1 | findstr /C:"Python 3.7.8" > nul 2> nul || call :install_python
goto vs2015

:install_python
echo There is not Python in this PC, will install Python-3.7.8.
echo Download package from https://npm.taobao.org/mirrors/python/3.7.8/python-3.7.8-amd64.exe ...
wget -O python-3.7.8-amd64.exe https://npm.taobao.org/mirrors/python/3.7.8/python-3.7.8-amd64.exe
echo Install Python-3.7.8 ...
:: /passive [silent install]
:: InstallAllUsers [add path for all users]
:: PrependPath [add script/install into PATH]
:: TargetDir [install directory]
start /wait python-3.7.8-amd64.exe /passive InstallAllUsers=1 PrependPath=1 TargetDir=C:\Python37
if %errorlevel% == 0 (
  echo Install python-3.7.8 success!
) else (
  echo Error***** Install python-3.7.8 failed, please re-install it manually.
)
del python-3.7.8-amd64.exe
goto :eof
:: ===== end step 3: Python =====

:: ===== start step 4: Visual Studio 2015 =====
:: Download Visual Studio 2015 when it not installed.
:vs2015
echo ">>>>>>>> step [4/7]: Visual Studio 2015"
cmd /C "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64 > nul 2> nul || call :install_visual_studio
goto :cuda10

:install_visual_studio
echo There is not Visual Studio in this PC, will install VS2015.
echo Download package from "https://download.my.visualstudio.com/pr/en_visual_studio_professional_2015_with_update_3_x86_x64_web_installer_8922978.exe"
wget -O vs_installer.exe "https://download.my.visualstudio.com/pr/en_visual_studio_professional_2015_with_update_3_x86_x64_web_installer_8922978.exe?t=9ee7a96d-ca80-4b84-af2c-7dd86996a0aa&e=1600103404&h=3cdea1e81c04aa4e846f5314972c46eb&su=1"
echo Install Visual Studio 2015 ...
:: /passive [silent install]
:: /norestart [no restart]
:: /NoRefresh [no refresh]
:: /InstallSelectableItems NativeLanguageSupport_Group [select Visual C++ for installing]
start /wait visual_installer.exe /passive /norestart /NoRefresh /InstallSelectableItems NativeLanguageSupport_Group
if %errorlevel% == 0 (
  echo Install Visual Studio 2015 success!
) else (
  echo Error***** Install Visual Studio 2015 failed, please re-install it manually.
)
del vs_installer.exe
goto :eof
:: ===== end step 4: Visual Studio 2015 =====

:: ===== start step 5: CUDA 10 =====
:cuda10
echo ">>>>>>>> step [5/7]: CUDA 10.0"
nvcc --version > nul 2> nul || call :install_cuda
goto java-jre

:install_cuda
echo There is not CUDA in this PC, will install CUDA-10.0.
echo Download package from "https://developer.download.nvidia.cn/compute/cuda/10.0/secure/Prod/network_installers/cuda_10.0.130_win10_network.exe"
wget -O cuda_installer.exe "https://developer.download.nvidia.cn/compute/cuda/10.0/secure/Prod/network_installers/cuda_10.0.130_win10_network.exe?hG7oBtA2CnxZG7d39onmBdtzrIa2cOukrmW8I0qk3h36vb2Sj0yYGjMElJlxlNhjx8Xu5RlbmdBhCWvP2QcEqMjCoKCXe5lOgr5uIIso_7LqrotgQHbZRZSVBYRT4bIAHPVSPrr4_4KczKvI9Nf3mbO9RJ2Vj6ECD5QphRMJBus0KKNVxO1gsplVL5qaCnE"
echo Install CUDA-10.0 ...
:: -s [silent install]
start /wait cuda_installer.exe -s
if %errorlevel% == 0 (
  echo Install CUDA-10.0 success!
) else (
  echo Error***** Install CUDA-10.0 failed, please re-install it manually.
)
del cuda_installer.exe
goto :eof
:: ===== end step 5: CUDA 10 =====

:: ===== start step 6: java jre =====
:java-jre
echo ">>>>>>>> step [6/7]: java jre"
goto xly-agent
:: ===== end step 6: java jre =====

:: ===== start step 7: xly agent =====
:xly-agent
echo ">>>>>>>> step [7/7]: xly agent"
goto :eof
:: ===== end step 8: xly agent =====
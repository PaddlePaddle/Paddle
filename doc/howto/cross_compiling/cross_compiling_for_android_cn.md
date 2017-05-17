# 构建Android平台上的PaddlePaddle库

用户可通过交叉编译的方式，在用户熟悉的开发平台（Linux，Mac OS X和Windows）上编译Android平台上适用的PaddlePaddle库。
本文档将以Linux x86-64平台为例，介绍交叉编译Android平台上适用的PaddlePaddle库的方法和步骤。

## 准备交叉编译环境

从源码交叉编译PaddlePaddle，用户需要提前准备好交叉编译环境。Android平台上使用的C/C++交叉编译工具链为[Android NDK](https://developer.android.com/ndk/downloads/index.html?hl=zh-cn)，用户可自行前往下载预编译好的版本，也可通过以下命令获取：

```bash
wget -q https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
unzip -q android-ndk-r14b-linux-x86_64.zip
```

Android NDK中包含了所有Android API级别、所有架构（arm/arm64/x86/mips）需要用到的编译工具和系统库。用户可根据自己的编译目标架构、所需支持的最低Android API级别，构建[独立工具链](https://developer.android.google.cn/ndk/guides/standalone_toolchain.html?hl=zh-cn)。
比如：

```bash
your/path/to/android-ndk-r14b-linux-x86_64/build/tools/make-standalone-toolchain.sh \
        --arch=arm --platform=android-21 --install-dir=your/path/to/my_standalone_toolchain
```

此命令将在your/path/to/my_standalone_toolchain目录生成一套编译工具链，面向架构为32位ARM架构，支持的最小的Android API级别为21，使用的编译器为arm-linux-androideabi-gcc (GCC) 4.9。

注意：**PaddlePaddle要求使用的编译工具链所支持的Andoid API级别不小于21**。

## 配置交叉编译参数

CMake系统对交叉编译提供了支持[cmake-toolchains](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling)。为了简化cmake配置，PaddlePaddle为交叉编译提供了工具链配置文档[cmake/cross_compiling/android.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/android.cmake)，以提供一些默认的编译器和编译参数相关配置。注意，从CMake 3.7版本开始，CMake官方对Android平台的交叉编译提供了通用的支持。PaddlePaddle若检测到用户使用的CMake版本不低于3.7时，将会将用户传进来的配置参数传递CMake系统，交由CMake系统本身来处理。有关参数配置的详细说明见[cmake-toolchains](https://cmake.org/cmake/help/v3.7/manual/cmake-toolchains.7.html#cross-compiling)。

交叉编译Android版本的PaddlePaddle库时，有一些必须配置的参数：
- `CMAKE_SYSTEM_NAME`，CMake编译的目标平台，必须设置为`Android`。在设置`CMAKE_SYSTEM_NAME=Android`后，PaddlePaddle的CMake系统才认为是在交叉编译Android系统的版本，并自动编译宿主机版protoc可执行文件、目标机版protobuf库、以及Android所需`arm_soft_fp_abi`分支的目标机版OpenBLAS库。此外，还会强制设置一些PaddlePaddle参数的值（`WITH_GPU=OFF`、`WITH_AVX=OFF`、`WITH_PYTHON=OFF`、`WITH_RDMA=OFF`）。
- `WITH_C_API`，必须设置为`ON`。在Android平台上只支持使用C-API来预测。
- `WITH_SWIG_PY`，必须设置为`OFF`。在Android平台上不支持通过swig调用来训练或者预测。

Android平台可选配置参数：

- `ANDROID_STANDALONE_TOOLCHAIN`，独立工具链所在的绝对路径，或者相对于构建目录的相对路径。PaddlePaddle的CMake系统将根据该值自动推导和设置需要使用的交叉编译器、sysroot、以及Android API级别；否则，用户需要在cmake时手动设置这些值。无默认值。
- `ANDROID_ABI`，目标架构ABI。目前只支持`armeabi-v7a`，默认值为`armeabi-v7a`。
- `ANDROID_NATIVE_API_LEVEL`，工具链的Android API级别。若没有显式设置，PaddlePaddle将根据`ANDROID_STANDALONE_TOOLCHAIN`的值自动推导得到。
- `ANROID_ARM_MODE`，是否使用ARM模式。可设置`ON/OFF`，默认值为`ON`。
- `ANDROID_ARM_NEON`，是否使用NEON指令。目前必须设置成`ON`，默认值为`ON`。

其他配置参数：

- `HOST_C/CXX_COMPILER`，宿主机的C/C++编译器。在编译宿主机版protoc可执行文件和目标机版OpenBLAS库时需要用到。默认设置成环境变量`CC`的值；若环境变量`CC`没有设置，则设置成`cc`编译器。

一种常用的cmake配置如下：

```bash
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=your/path/to/my_standalone_toolchain \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
```

用户还可根据自己的需求设置其他编译参数。比如希望最小化生成的库的大小，可以设置`CMAKE_BUILD_TYPE`为`MinSizeRel`；若希望最快的执行速度，则可设置`CMAKE_BUILD_TYPE`为`Release`。亦可以通过手动设置`CMAKE_C/CXX_FLAGS_MINSIZEREL/RELEASE`来影响PaddlePaddle的编译过程。

## 编译和安装

CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle预测库。

```bash
make
make install
```

注意：如果你曾经在源码目录下编译过其他平台的PaddlePaddle库，请先使用`rm -rf`命令删除`third_party`目录和`build`目录，以确保所有的第三方依赖库和PaddlePaddle代码都是针对新的CMake配置重新编译的。

执行完安装命令后，`your/path/to/install`目录中会包含`include`和`lib`目录，其中`include`中包含C-API的头文件，`lib`中包含一个Android版本的库。自此，PaddlePaddle的已经安装完成，用户可将`your/path/to/install`目录下的生成文件用于深度学习相关Android App中，调用方法见C-API文档。

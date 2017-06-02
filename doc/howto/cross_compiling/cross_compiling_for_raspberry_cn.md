# 构建Raspberry Pi平台上的PaddlePaddle库

对于Rasspberry Pi系统，用户可通过ssh等方式登录到Raspberry Pi系统上，按照[源码编译PaddlePaddle](http://www.paddlepaddle.org/doc_cn/getstarted/build_and_install/cmake/build_from_source_cn.html)相关文档所述，直接编译Raspberry Pi平台上适用的PaddlePaddle库。

用户也可以在自己熟悉的开发平台上，通过交叉编译的方式来编译。这篇文档将以Linux x86-64平台为例，介绍交叉编译Raspberry Pi平台上适用的PaddlePaddle的方法和步骤。

## 准备交叉编译环境

从源码交叉编译PaddlePaddle，用户需要提前准备好交叉编译环境。用户可自行前往[github](https://github.com/raspberrypi/tools)下载Raspberry Pi平台使用的C/C++交叉编译工具链，也可通过以下命令获取：

```bash
git clone https://github.com/raspberrypi/tools.git
```

该github仓库中包含若干个预编译好的、针对不同平台的编译工具。宿主机是Linux x86-64环境，则需选用`arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64`下的作为编译工具，所使用的编译器为arm-linux-gnueabihf-gcc 4.8.3。

注意，该编译工具链需要系统glibc支持2.14以上。

## 配置交叉编译参数

CMake系统对交叉编译提供了支持[cmake-toolchains](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling)。为了简化cmake配置，PaddlePaddle为交叉编译提供了工具链配置文档[cmake/cross_compiling/raspberry_pi.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/raspberry_pi.cmake)，以提供一些默认的编译器和编译参数相关配置。

交叉编译Raspberry Pi版本PaddlePaddle库时，有一些必须配置的参数：

- `CMAKE_SYSTEM_NAME`，CMake编译的目标平台，必须配置为`RPi`。在设置`CMAKE_SYSTEM_NAME=RPi`后，PaddlePaddle的CMake系统才认为在是在交叉编译Raspberry Pi系统的版本，并自动编译宿主机版protoc可执行文件、目标机版protobuf库、以及目标机版OpenBLAS库。

Raspberry Pi平台可选配置参数：

- `RPI_TOOLCHAIN`，编译工具链所在的绝对路径，或者相对于构建目录的相对路径。PaddlePaddle的CMake系统将根据该值自动设置需要使用的交叉编译器；否则，用户需要在cmake时手动设置这些值。无默认值。
- `RPI_ARM_NEON`，是否使用NEON指令。目前必须设置成`ON`，默认值为`ON`。

其他配置参数：

- `HOST_C/CXX_COMPILER`，宿主机的C/C++编译器。在编译宿主机版protoc可执行文件和目标机版OpenBLAS库时需要用到。默认设置成环境变量`CC`的值；若环境变量`CC`没有设置，则设置成`cc`编译器。

cmake参数如下；

```
cmake -DCMAKE_SYSTEM_NAME=RPi \
      -DRPI_TOOLCHAIN=your/path/to/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64 \
      -DRPI_ARM_NEON=ON \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_GPU=OFF \
      -DWITH_C_API=ON \
      -DWITH_PYTHON=OFF \
      -DWITH_SWIG_PY=OFF \
      ..
```

用户还可根据自己的需求设置其他编译参数。比如希望最小化生成的库的大小，可以设置`CMAKE_BUILD_TYPE`为`MinSizeRel`；若希望最快的执行速度，则可设置`CMAKE_BUILD_TYPE`为`Release`。亦可以通过手动设置`CMAKE_C/CXX_FLAGS_MINSIZEREL/RELEASE`来影响PaddlePaddle的编译过程。

## 编译和安装

CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle。

```bash
make
make install
```

注意：如果你曾经在源码目录下编译过其他平台的PaddlePaddle库，请先使用`rm -rf`命令删除`third_party`目录和`build`目录，以确保所有的第三方依赖库和PaddlePaddle代码都是针对新的CMake配置重新编译的。

执行完安装命令后，由于上一步cmake配置中`WITH_C_API`设置为`ON`，`your/path/to/install`目录中会包含`include`和`lib`目录，其中`include`中包含C-API的头文件，`lib`中包含一个Raspberry Pi版本的库。

更多的编译配置见[源码编译PaddlePaddle](http://www.paddlepaddle.org/doc_cn/getstarted/build_and_install/cmake/build_from_source_cn.html)相关文档。

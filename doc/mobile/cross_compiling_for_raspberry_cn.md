# Raspberry Pi平台编译指南

通常有两个方法来构建基于 Rasspberry Pi 的版本：

1. 通过ssh等方式登录到Raspberry Pi系统上来构建。所需的开发工具和第三方库可以参考 [`/Dockerfile`](https://github.com/PaddlePaddle/Paddle/blob/develop/Dockerfile)。

1. 另一个方法是交叉编译。这篇文档介绍在 Linux/x64 上交叉编译Raspberry Pi平台上适用的PaddlePaddle的方法和步骤。

## 安装交叉编译器

克隆下面 Github repo

```bash
git clone https://github.com/raspberrypi/tools.git
```

即可在 `./tools/tree/master/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64` 目录里找到交叉编译器 arm-linux-gnueabihf-gcc 4.8.3。运行该编译工具链需要一台 Linux x64 机器上以及 2.14版本以上的 glibc。

## 配置交叉编译参数

CMake[支持交叉编译](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling)。PaddlePaddle for Raspberry Pi的配置信息在[cmake/cross_compiling/raspberry_pi.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/raspberry_pi.cmake)。

交叉编译Raspberry Pi版本PaddlePaddle库时，有一些必须配置的参数：

- `CMAKE_SYSTEM_NAME`：CMake编译的目标平台，必须配置为`RPi`。在设置`CMAKE_SYSTEM_NAME=RPi`后，PaddlePaddle的CMake系统才认为在是在交叉编译Raspberry Pi系统的版本，并自动编译宿主机版protoc可执行文件、目标机版protobuf库、以及目标机版OpenBLAS库。

- `RPI_TOOLCHAIN`：编译工具链所在的绝对路径，或者相对于构建目录的相对路径。PaddlePaddle的CMake系统将根据该值自动设置需要使用的交叉编译器；否则，用户需要在cmake时手动设置这些值。无默认值。

- `RPI_ARM_NEON`：是否使用NEON指令。目前必须设置成`ON`，默认值为`ON`。

- `HOST_C/CXX_COMPILER`，宿主机的C/C++编译器。在编译宿主机版protoc可执行文件和目标机版OpenBLAS库时需要用到。默认设置成环境变量`CC`的值；若环境变量`CC`没有设置，则设置成`cc`编译器。

一个常用的CMake配置如下：

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

其中`WITH_C_API=ON`表示需要构建推理库。

用户还可根据自己的需求设置其他编译参数。比如希望最小化生成的库的大小，可以设置`CMAKE_BUILD_TYPE`为`MinSizeRel`；若希望最快的执行速度，则可设置`CMAKE_BUILD_TYPE`为`Release`。

## 编译和安装

CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle。

```bash
make
make install
```

注意：如果你曾经在源码目录下编译过其他平台的PaddlePaddle库，请先使用`rm -rf`命令删除`third_party`目录和`build`目录，以确保所有的第三方依赖库和PaddlePaddle代码都是针对新的CMake配置重新编译的。

执行完安装命令后，`your/path/to/install`目录中会包含`include`和`lib`目录，其中`include`中包含C-API的头文件，`lib`中包含一个Raspberry Pi版本的库。

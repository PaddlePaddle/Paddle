# Android平台编译指南

用户可通过如下两种方式，交叉编译Android平台上适用的PaddlePaddle库：

- [基于Docker容器的编译方式](#基于docker容器的编译方式)
- [基于Linux交叉编译环境的编译方式](#基于linux交叉编译环境的编译方式)

## 基于Docker容器的编译方式
Docker能在所有主要操作系统（包括Linux，Mac OS X和Windows）上运行，因此，使用基于Docker容器的编译方式，用户可在自己熟悉的开发平台上编译Android平台上适用的PaddlePaddle库。

### 构建PaddlePaddle的Android开发镜像
我们把PaddlePaddle的交叉编译环境打包成一个镜像，称为开发镜像，里面涵盖了交叉编译Android版PaddlePaddle库需要的所有编译工具。

```bash
$ git clone https://github.com/PaddlePaddle/Paddle.git
$ cd Paddle
$ docker build -t username/paddle-android:dev . -f Dockerfile.android
```

用户也可以使用PaddlePaddle提供的官方开发镜像：

```bash
$ docker pull paddlepaddle/paddle:latest-dev-android
```

对于国内用户，我们提供了加速访问的镜像源：

```bash
$ docker pull docker.paddlepaddlehub.com/paddle:latest-dev-android
```

### 编译PaddlePaddle C-API库
构建好开发镜像后，即可使用开发镜像来编译Android版PaddlePaddle C-API库。
Android的Docker开发镜像向用户提供两个可配置的参数：

<table class="docutils">
<colgroup>
  <col width="25%" />
  <col width="50%" />
  <col width="25%" />
</colgroup>
<thead valign="bottom">
  <tr class="row-odd">
  <th class="head">Argument</th>
  <th class="head">Optional Values</th>
  <th class="head">Default</th>
</tr>
</thead>
<tbody valign="top">
  <tr class="row-even">
  <td>ANDROID_ABI</td>
  <td>armeabi-v7a, arm64-v8a</td>
  <td>armeabi-v7a</td>
</tr>
<tr class="row-odd">
  <td>ANDROID_API</td>
  <td>>= 16</td>
  <td>21</td>
</tr>
</tbody>
</table>

- 编译`armeabi-v7a`，`Android API 21`的PaddlePaddle库

```bash
$ docker run -it --rm -v $PWD:/paddle -w /paddle -e "ANDROID_ABI=armeabi-v7a" -e "ANDROID_API=21" username/paddle-android:dev ./paddle/scripts/paddle_build.sh build_android
```

- 编译`arm64-v8a`，`Android API 21`的PaddlePaddle库

```bash
$ docker run -it --rm -v $PWD:/paddle -w /paddle -e "ANDROID_ABI=arm64-v8a" -e "ANDROID_API=21" username/paddle-android:dev ./paddle/scripts/paddle_build.sh build_android
```

执行上述`docker run`命令时，容器执行[paddle/scripts/paddle_build.sh build_android](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh)脚本。该脚本中记录了交叉编译Android版PaddlePaddle库常用的CMake配置，并且会根据`ANDROID_ABI`和`ANDROID_API`自动构建独立工具链、进行编译和安装。由于arm64架构要求Android API不小于21。因此当`ANDROID_ABI=arm64-v8a`，`ANDROID_API<21`时，Docker容器中将默认使用`Android API 21`的编译工具链。用户可以参考下文[配置交叉编译参数](#配置交叉编译参数)章节，根据个人的需求修改定制Docker容器所执行的脚本。编译安装结束之后，PaddlePaddle的C-API库将被安装到`$PWD/install_android`目录，所依赖的第三方库同时也被安装到`$PWD/install_android/third_party`目录。

## 基于Linux交叉编译环境的编译方式
本文档将以Linux x86-64平台为例，介绍交叉编译Android平台上适用的PaddlePaddle库的方法和步骤。

### 准备交叉编译环境

从源码交叉编译PaddlePaddle，用户需要提前准备好交叉编译环境。Android平台上使用的C/C++交叉编译工具链为[Android NDK](https://developer.android.com/ndk/downloads/index.html?hl=zh-cn)，用户可自行前往下载预编译好的版本，也可通过以下命令获取：

```bash
wget -q https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
unzip -q android-ndk-r14b-linux-x86_64.zip
```

Android NDK中包含了所有Android API级别、所有架构（arm/arm64/x86/mips）需要用到的编译工具和系统库。用户可根据自己的编译目标架构、所需支持的最低Android API级别，构建[独立工具链](https://developer.android.google.cn/ndk/guides/standalone_toolchain.html?hl=zh-cn)。

- 构建`armeabi-v7a`、 `Android API 21`的独立工具链：

```bash
your/path/to/android-ndk-r14b-linux-x86_64/build/tools/make-standalone-toolchain.sh \
        --arch=arm --platform=android-21 --install-dir=your/path/to/arm_standalone_toolchain
```

此命令将在`your/path/to/arm_standalone_toolchain`目录生成一套独立编译工具链，面向架构为32位ARM架构，支持的最小的Android API级别为21，支持编译器`arm-linux-androideabi-gcc (GCC) 4.9`和`clang 3.8`。

- 构建`arm64-v8a`、 `Android API 21`的独立工具链：

```bash
your/path/to/android-ndk-r14b-linux-x86_64/build/tools/make-standalone-toolchain.sh \
        --arch=arm64 --platform=android-21 --install-dir=your/path/to/arm64_standalone_toolchain
```

此命令将在`your/path/to/arm64_standalone_toolchain`目录生成一套独立编译工具链，面向架构为64位ARM64架构，支持的最小Android API级别为21，支持编译器`arm-linux-androideabi-gcc (GCC) 4.9`和`clang 3.8`。

### 配置交叉编译参数

CMake系统对交叉编译提供了支持[cmake-toolchains](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling)。为了简化cmake配置，PaddlePaddle为交叉编译提供了工具链配置文档[cmake/cross_compiling/android.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/android.cmake)，以提供一些默认的编译器和编译参数相关配置。注意，从CMake 3.7版本开始，CMake官方对Android平台的交叉编译提供了通用的支持。PaddlePaddle若检测到用户使用的CMake版本不低于3.7时，将会将用户传进来的配置参数传递CMake系统，交由CMake系统本身来处理。有关参数配置的详细说明见[cmake-toolchains](https://cmake.org/cmake/help/v3.7/manual/cmake-toolchains.7.html#cross-compiling)。

交叉编译Android版本的PaddlePaddle库时，有一些必须配置的参数：
- `CMAKE_SYSTEM_NAME`，CMake编译的目标平台，必须设置为`Android`。在设置`CMAKE_SYSTEM_NAME=Android`后，PaddlePaddle的CMake系统才认为是在交叉编译Android系统的版本，并自动编译PaddlePaddle所需的所有第三方库。此外，还会强制设置一些PaddlePaddle参数的值（`WITH_GPU=OFF`、`WITH_AVX=OFF`、`WITH_PYTHON=OFF`、`WITH_RDMA=OFF`、`WITH_MKL=OFF`、`WITH_GOLANG=OFF`）。
- `WITH_C_API`，必须设置为`ON`。在Android平台上只支持使用C-API来预测。
- `WITH_SWIG_PY`，必须设置为`OFF`。在Android平台上不支持通过swig调用来训练或者预测。

Android平台可选配置参数：

- `ANDROID_STANDALONE_TOOLCHAIN`，独立工具链所在的绝对路径，或者相对于构建目录的相对路径。PaddlePaddle的CMake系统将根据该值自动推导和设置需要使用的交叉编译器、sysroot、以及Android API级别；否则，用户需要在cmake时手动设置这些值。无默认值。
- `ANDROID_TOOLCHAIN`，目标工具链。可设置`gcc/clang`，默认值为`clang`。
	- CMake 3.7以上，将会始终使用`clang`工具链；CMake 3.7以下，可设置`ANDROID_TOOLCHAIN=gcc`以使用`gcc`工具链。
	- Android官方提供的`clang`编译器要求系统支持`GLIBC 2.15`以上。
- `ANDROID_ABI`，目标架构ABI。目前支持`armeabi-v7a`和`arm64-v8a`，默认值为`armeabi-v7a`。
- `ANDROID_NATIVE_API_LEVEL`，工具链的Android API级别。若没有显式设置，PaddlePaddle将根据`ANDROID_STANDALONE_TOOLCHAIN`的值自动推导得到。
- `ANROID_ARM_MODE`，是否使用ARM模式。
	- `ANDROID_ABI=armeabi-v7a`时，可设置`ON/OFF`，默认值为`ON`；
	- `ANDROID_ABI=arm64-v8a`时，不需要设置。
- `ANDROID_ARM_NEON`，是否使用NEON指令。
	- `ANDROID_ABI=armeabi-v7a`时，可设置`ON/OFF`，默认值为`ON`；
	- `ANDROID_ABI=arm64-v8a`时，不需要设置。

其他配置参数：

- `USE_EIGEN_FOR_BLAS`，是否使用Eigen库进行矩阵计算。可设置`ON/OFF`，默认值为`OFF`。
- `HOST_C/CXX_COMPILER`，宿主机的C/C++编译器。在编译宿主机版protoc可执行文件和目标机版OpenBLAS库时需要用到。默认设置成环境变量`CC/CXX`的值；若环境变量`CC/CXX`没有设置，则设置成`cc/c++`编译器。

常用的cmake配置如下：

```bash
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=your/path/to/arm_standalone_toolchain \
      -DANDROID_ABI=armeabi-v7a \
      -DANDROID_ARM_NEON=ON \
      -DANDROID_ARM_MODE=ON \
      -DUSE_EIGEN_FOR_BLAS=ON \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
```

```
cmake -DCMAKE_SYSTEM_NAME=Android \
      -DANDROID_STANDALONE_TOOLCHAIN=your/path/to/arm64_standalone_toolchain \
      -DANDROID_ABI=arm64-v8a \
      -DUSE_EIGEN_FOR_BLAS=OFF \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      ..
```

用户还可根据自己的需求设置其他编译参数。

- 设置`CMAKE_BUILD_TYPE`为`MinSizeRel`，最小化生成的库的大小。
- 设置`CMAKE_BUILD_TYPE`为`Release`，获得最快的执行速度，
- 用户亦可以通过手动设置`CMAKE_C/CXX_FLAGS`来影响PaddlePaddle的编译过程。

**性能TIPS**，为了达到最快的计算速度，在CMake参数配置上，有以下建议：

- 设置`CMAKE_BUILD_TYPE`为`Release`
- 使用`clang`编译工具链
- `armeabi-v7a`时，设置`USE_EIGEN_BLAS=ON`，使用Eigen进行矩阵计算；`arm64-v8a`时，设置`USE_EIGEN_FOR_BLAS=OFF`，使用OpenBLAS进行矩阵计算

### 编译和安装

CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle预测库。

```bash
make
make install
```

注意：如果你曾经在源码目录下编译过其他平台的PaddlePaddle库，请先使用`rm -rf`命令删除`third_party`目录和`build`目录，以确保所有的第三方依赖库和PaddlePaddle代码都是针对新的CMake配置重新编译的。

执行完安装命令后，`your/path/to/install`目录中会包含`include`、`lib`和`third_party`目录，其中`include`中包含C-API的头文件，`lib`中包含若干个不同Android ABI的PaddlePaddle库，`third_party`中包含所依赖的所有第三方库。自此，PaddlePaddle的已经安装完成，用户可将`your/path/to/install`目录下的生成文件用于深度学习相关Android App中，调用方法见C-API文档。

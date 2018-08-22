# Anakin安装指南 #

本文档介绍如何源码编译安装 Anakin。开始前，请确认已备有 Linux 操作系统的计算机。

## 1、环境要求 ##

*  CentOS 或 Ubuntu
*  GNU Make: 3.81 +
*  CMake: 2.8.12 +
*  GCC / G++ / C++: 4.8 ~ 5.4
*  Protobuf: 3.1.0 +

## 2、安装步骤 ##

本节将叙述 Anakin 在 Nvidia GPU、Intel CPU 和 AMD GPU 上的安装步骤。Anakin 移动版安装请参考 [ARM 安装指南](run_on_arm_ch.md)，我们将在后续版本提供寒武纪和比特大陆的解决方案。

### 2.1、准备 ###

首先，请获取 Anakin 开发分支的源码。

    git clone -b developing https://github.com/PaddlePaddle/Anakin.git  

如果您需要安装 Protobuf，请运行下列命令。

    wget https://github.com/google/protobuf/releases/download/v3.1.0/protobuf-cpp-3.1.0.tar.gz  
    tar -zxvf protobuf-cpp-3.1.0.tar.gz
    cd protobuf-3.1.0
    ./autogen.sh
    ./configure
    make && make install

注意：上述命令可能需要开发者权限，并覆盖系统原有的 Protobuf。若您需解决多版本共存问题，请在上面的 `./configure` 后使用 `--prefix` 后缀，并在 `$HOME/.bashrc` 修改环境变量，直至可以使用下列命令。

    $ protoc --version
    libprotoc 3.1.0

此外，您还可能需要设置 cmake 的 `CMAKE_PREFIX_PATH` 选项，以告知其 Protobuf 链接库和头文件的路径。

#### Anakin - NVGPU ###

除了环境要求中列举的部分，Anakin - NVGPU 版本还依赖 [CUDA 8.0 +](https://developer.nvidia.com/cuda-zone) 和 [cuDNN 7.0 +](https://developer.nvidia.com/cudnn)，请您注意版本的匹配。


#### Anakin - CPU ###

在编译 CPU 版本前，我们建议您升级 GCC-G++ 至 5.4.0 以上，链接的 libm.so 库版本高于 2.17，以发挥更佳性能。

#### Anakin - AMDGPU ###

在编译 AMD GPU 版本之前，需要首先安装 [ROCm 驱动程序](https://github.com/RadeonOpenCompute/ROCm/blob/master/README.md)。

### 2.2、安装 ###

我们将不同架构的安装命令写成了脚本，请在准备工作完成之后直接运行。

#### Anakin - NVGPU ###

    $ sh Anakin/tools/gpu_build.sh

#### Anakin - CPU ###

    $ sh Anakin/tools/x86_build.sh

#### Anakin - AMDGPU ###

    $ sh Anakin/tools/amd_gpu_build.sh

### 2.3、离线安装指南 ###

在一些环境下可能需要离线方式安装 Anakin，下面以 CentOS 6.3 为例进行说明。

#### 2.3.1、安装包准备 ###

将 Anakin 源码和 `anakin_third_party.tar.gz` 拷贝到离线环境中。

    wget https://github.com/PaddlePaddle/Anakin/archive/developing.zip

除 GNU Make 3.81、CMake 2.8.12、GCC-G++ 4.8.2 和上一小节所述不同架构的依赖外，您可能还需要准备 [AutoConf 2.63](https://centos.pkgs.org/6/centos-i386/autoconf-2.63-5.1.el6.noarch.rpm.html)、[AutoMake 1.11](https://centos.pkgs.org/6/centos-i386/automake-1.11.1-4.el6.noarch.rpm.html)、[libtool 2.26](https://centos.pkgs.org/6/centos-x86_64/libtool-2.2.6-15.5.el6.x86_64.rpm.html)。

#### 2.3.2、离线安装 ###

将 `anakin_third_party.tar.gz` 解压至 Anakin/third_party，并按上一小节的说明安装 Protobuf。

注释 `cmake/external/mkldnn.cmake`、`cmake/external/mklyl.cmake` 和 `cmake/external/xbyak.cmake` 的 `ExternalProject_Add` 语句。

最后执行 `tools/` 中的脚本，进行安装。

## 3、其它 ##

1、遇到任何问题，请提 [Issue](https://github.com/PaddlePaddle/Anakin/issues) 或邮件至 Anakin@baidu.com；  
2、模型解析器使用说明请移步 [这里](Converter_ch.md)。

## 从源码编译安装Anakin ##

我们已经在CentOS 7.3上成功的安装和测试了Anakin，对于其他操作系统，我们将很快支持。

### 安装概览 ###

* [在CentOS上安装 Anakin]()
* [在Ubuntu上安装 Anakin]()
* [在ARM上安装 Anakin](run_on_arm_ch.md)
* [验证安装]()


### 在CentOS上安装 Anakin ###
#### 1. 系统要求 ####

*  make 3.82+
*  cmake 2.8.12+
*  gcc 4.8.2+
*  g++ 4.8.2+
*  其他需要补充的。。。

#### 2. 编译CPU版Anakin ####

暂时不支持

#### 3. 编译支持NVIDIA GPU的Anakin ####

- 3.1. 安装依赖
  - 3.1.1 protobuf  
    >$ git clone https://github.com/google/protobuf  
    >$ cd protobuf  
    >$ git submodule update --init --recursive  
    >$ ./autogen.sh  
    >$ ./configure --prefix=/path/to/your/insall_dir  
    >$ make  
    >$ make check  
    >$ make install  
    >$ sudo ldconfig


    如安装protobuf遇到任何问题，请访问[这里](https://github.com/google/protobuf/blob/master/src/README.md)

- 3.2 CUDA Toolkit
  - [CUDA 8.0](https://developer.nvidia.com/cuda-zone) or higher. 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
  - [cuDNN v7](https://developer.nvidia.com/cudnn). 具体信息参见[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). 
- 3.3  编译Anakin
  >$ git clone https:/xxxxx  
  >$ cd anakin  
  >$ mkdir build  
  >$ camke ..  
  >$ make


#### 4. 编译支持AMD GPU的Anakin ####

暂时还不支持


### 在Ubuntu上安装 Anakin ###

暂时还不支持


### 在ARM上安装 Anakin ###

暂时还不支持

### 验证安装 ###
we are coming soon...

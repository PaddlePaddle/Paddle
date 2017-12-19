## 编译 PaddlePaddle 链接库

### 概述

使用 C-API 进行预测依赖于将 PaddlePaddle 核心代码编译成链接库，只需在编译时指定编译选项：`-DWITH_C_API=ON`。同时，**建议将：`DWITH_PYTHON`，`DWITH_SWIG_PY`，`DWITH_GOLANG`，均设置为`OFF`**，以避免链接不必要的库。其它编译选项按需进行设定。

```shell
INSTALL_PREFIX=/path/of/capi/
PADDLE_ROOT=/path/of/paddle_source/
cmake $PADDLE_ROOT -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
      -DCMAKE_BUILD_TYPE=Release \
      -DWITH_C_API=ON \
      -DWITH_SWIG_PY=OFF \
      -DWITH_GOLANG=OFF \
      -DWITH_PYTHON=OFF \
      -DWITH_MKLML=OFF \
      -DWITH_MKLDNN=OFF \
      -DWITH_GPU=OFF \
      ...
```
在上面的代码片段中，`PADDLE_ROOT` 表示 PaddlePaddle 源码所在目录，生成Makefile文件后执行：`make && make install`。成功执行后，使用CAPI所需的依赖（包括：（1）编译出的PaddlePaddle 链接和头文件；（2）第三方链接库和头文件）均会存放于`INSTALL_PREFIX`目录中。

编译成功后在 `INSTALL_PREFIX` 下会看到如下目录结构（包括了编译出的PaddlePaddle头文件和链接库，以及第三方依赖链接库和头文件（如果需要，由链接方式决定））：

```text
├── include
│   └── paddle
│       ├── arguments.h
│       ├── capi.h
│       ├── capi_private.h
│       ├── config.h
│       ├── error.h
│       ├── gradient_machine.h
│       ├── main.h
│       ├── matrix.h
│       ├── paddle_capi.map
│       └── vector.h
├── lib
│   ├── libpaddle_capi_engine.a
│   ├── libpaddle_capi_layers.a
│   ├── libpaddle_capi_shared.dylib
│   └── libpaddle_capi_whole.a
└── third_party
    ├── ......
```

### 链接方式说明

目前提供三种链接方式：

1. 链接`libpaddle_capi_shared.so` 动态库
    - 使用 PaddlePaddle C-API 开发预测程序链接`libpaddle_capi_shared.so`时，需注意：
        1. 如果编译时指定编译CPU版本，且使用`OpenBLAS`矩阵库，在使用CAPI开发预测程序时，只需要链接`libpaddle_capi_shared.so`这一个库。
        1. 如果是用编译时指定CPU版本，且使用`MKL`矩阵库，由于`MKL`库有自己独立的动态库文件，在使用PaddlePaddle CAPI开发预测程序时，需要自己链接MKL链接库。
        1. 如果编译时指定编译GPU版本，CUDA相关库会在预测程序运行时动态装载，需要将CUDA相关的库设置到`LD_LIBRARY_PATH`环境变量中。
    - 这种方式最为简便，链接相对容易，**在无特殊需求情况下，推荐使用此方式**。

2. 链接静态库 `libpaddle_capi_whole.a`
    - 使用PaddlePaddle C-API 开发预测程序链接`libpaddle_capi_whole.a`时，需注意：
        1. 需要指定`-Wl,--whole-archive`链接选项。
        1. 需要显式地链接 `gflags`、`glog`、`libz`、`protobuf` 等第三方库，可在`INSTALL_PREFIX\third_party`下找到。
        1. 如果在编译 C-API 时使用OpenBLAS矩阵库，需要显示地链接`libopenblas.a`。
        1. 如果在编译 C-API 是使用 MKL 矩阵库，需要显示地链接 MKL 的动态库。

3. 链接静态库 `libpaddle_capi_layers.a`和`libpaddle_capi_engine.a`
    - 使用PaddlePaddle C-API 开发预测程序链接`libpaddle_capi_whole.a`时，需注意：
        1. 这种链接方式主要用于移动端预测。
        1. 为了减少生成链接库的大小把`libpaddle_capi_whole.a`拆成以上两个静态链接库。
        1. 需指定`-Wl,--whole-archive -lpaddle_capi_layers` 和 `-Wl,--no-whole-archive -lpaddle_capi_engine` 进行链接。
        1. 第三方依赖库需要按照与方式2同样方法显示地进行链接。

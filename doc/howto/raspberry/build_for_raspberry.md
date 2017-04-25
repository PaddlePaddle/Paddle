# 如何构建Raspberry pi下运行的PaddlePaddle

这里考虑的是交叉编译方式，即在Linux-x86环境下构建Raspberry pi下可运行的PaddlePaddle。

## 下载交叉编译环境
```
git clone https://github.com/raspberrypi/tools
```
如果host是x86-64环境，选用`arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64`下的作为编译工具。注意，需要系统glibc支持2.14以上。


## 编译第三方库
cmake编译PaddlePaddle时候会自动下载编译依赖的第三方库，不过openblas和protobuf最好还是在编译PaddlePaddle之前先编译好，这样可以保证编译PaddlePaddle的时候更加顺畅。

### 编译OpenBLAS
```
git clone https://github.com/xianyi/OpenBLAS.git
make TARGET=ARMV7 HOSTCC=gcc CC=arm-linux-gnueabihf-gcc NOFORTRAN=1 USE_THREAD=0
```

### 编译protobuf
```
git clone https://github.com/google/protobuf.git
git checkout 9f75c5aa851cd877fb0d93ccc31b8567a6706546
cmake ../protobuf/cmake \
-Dprotobuf_BUILD_TESTS=OFF \
-DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
-DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_LIBDIR=lib
```
注意：这样编译出来的`libprotobuf.a`和`protoc`都是ARM版本的，而我们需要的是一个x86-64版本的`protoc`，所以需要用host gcc再编译一遍protobuf然后使用其中的`protoc`。


## 编译Paddle
```
cmake .. -DWITH_GPU=OFF -DWITH_PYTHON=OFF -DWITH_SWIG_PY=OFF \
-DCMAKE_CXX_COMPILER:FILEPATH=arm-linux-gnueabihf-g++ \
-DCMAKE_C_COMPILER:FILEPATH=arm-linux-gnueabihf-gcc \
-DCMAKE_C_FLAGS="-mfpu=neon" \
-DCMAKE_CXX_FLAGS="-mfpu=neon" \
-DOPENBLAS_ROOT=openblas \
-DCMAKE_PREFIX_PATH=protobuf
```

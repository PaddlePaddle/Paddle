# Build PaddlePaddle for Raspberry Pi

We use one of the following two approaches to build the inference library of PaddlePaddle for Raspberry Pi:

1. Log in to a Raspberry Pi via SSH and build.  [`/Dockerfile`](https://github.com/PaddlePaddle/Paddle/blob/develop/Dockerfile) lists the required development tools and third-party dependencies.

1. An alternative is cross-compiling.  This article explains cross-compiling PaddlePaddle for Raspberry Pi on a Linux/x64 computer.

## The Cross-Compiling Toolchain

After cloning the following Github repo

```bash
git clone https://github.com/raspberrypi/tools.git
```

we could find the pre-built cross-compiler in `./tools/tree/master/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64`.  To run it on a Linux computer, we need glibc version >= 2.14.

## CMake Arguments

CMake supports [cross-compiling](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling).  We configure CMake arguments related with the cross-compiling for Raspberry Pi in [`cmake/cross_compiling/raspberry_pi.cmake`](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/raspberry_pi.cmake).

Some arguments you need to know:

- `CMAKE_SYSTEM_NAME`: The target platform.  Must be `RPi`.

- `RPI_TOOLCHAIN`: The absolute path of the cross-compiling toolchain.

- `RPI_ARM_NEON`: Use ARM NEON Intrinsics. Must be and default to `ON`.

- `HOST_C/CXX_COMPILER`: The C/C++ compiler for the host.  It is used to build building tools running on the host, for example, protoc.

A commonly-used CMake configuration is as follows:

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

The argument `WITH_C_API=ON` means to build the inference library.

Users can add more arguments.  For example, to minimize the size of the generated inference library, we can have `CMAKE_BUILD_TYPE=MinSizeRel`, for performance optimization, we can have `CMAKE_BUILD_TYPE=Release`.

## Build and Install

The following commands build the inference library of PaddlePaddle for Raspberry Pi and third-party dependencies.

```bash
make
make install
```

The intermediate files will be in `build`.  Third-party libraries will be in `build/third_party`.  If you have built for other platforms, e.g., Android or iOS, we might want to clear out these directories by running `rm -rf build`.

The infernece library will be in `your/path/to/install/lib`, with related header files in `your/path/to/install/include`.

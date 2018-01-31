# Build PaddlePaddle for Raspberry Pi

You may use any of the following two approaches to build the inference library of PaddlePaddle for Raspberry Pi:

1. Build using SSH: Log in to a Raspberry Pi using SSH and build the library. The required development tools and third-party dependencies are listed in here: [`/Dockerfile`](https://github.com/PaddlePaddle/Paddle/blob/develop/Dockerfile).

1. Cross-compile: We talk about how to cross-compile PaddlePaddle for Raspberry Pi on a Linux/x64 machine, in more detail in this article.

## The Cross-Compiling Toolchain

Step 1. Clone the Github repo by running the following command.

```bash
git clone https://github.com/raspberrypi/tools.git
```

Step 2. Use the pre-built cross-compiler found in `./tools/tree/master/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian-x64`.  To run it on a Linux computer, glibc version >= 2.14 is needed.

## CMake Arguments

CMake supports [cross-compiling](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling).  All CMake configuration arguments required for the cross-compilation for Raspberry Pi can be found in [`cmake/cross_compiling/raspberry_pi.cmake`](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/raspberry_pi.cmake).

Some important arguments that need to be set:

- `CMAKE_SYSTEM_NAME`: The target platform.  Must be `RPi`.

- `RPI_TOOLCHAIN`: The absolute path of the cross-compiling toolchain.

- `RPI_ARM_NEON`: Use ARM NEON Intrinsics. This is a required argument and set default to `ON`.

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

To build the inference library, please set the argument WITH\_C\_API to ON: `WITH_C_API=ON`.

You can add more arguments. For example, to minimize the size of the generated inference library, you may use `CMAKE_BUILD_TYPE=MinSizeRel`. For performance optimization, you may use `CMAKE_BUILD_TYPE=Release`.

## Build and Install

The following commands build the inference library of PaddlePaddle for Raspberry Pi and third-party dependencies.

```bash
make
make install
```

 The intermediate files will be stored in `build`. Third-party libraries will be located in `build/third_party`. If you have already built it for other platforms like Android or iOS, you may want to clear these directories by running the command: `rm -rf build`.

The infernece library will be in `your/path/to/install/lib`, with related header files in `your/path/to/install/include`.

# Build PaddlePaddle for Android

There are two approaches to build PaddlePaddle for Android: 

- [Cross-Compiling Using Docker](#cross-compiling-using-docker)
- [Cross-Compiling on Linux](#cross-compiling-on-linux) 

## Cross-Compiling Using Docker

Docker-based cross-compiling is the recommended approach because Docker runs on all major operating systems, including Linux, Mac OS X, and Windows.

### Build the Docker Image

The following steps pack all the tools that we need to build PaddlePaddle into a Docker image.

```bash
$ git clone https://github.com/PaddlePaddle/Paddle.git
$ cd Paddle
$ docker build -t paddle:dev-android . -f Dockerfile.android
```

Users can directly use the published Docker image.

```bash
$ docker pull paddlepaddle/paddle:latest-dev-android
```

For users in China, we provide a faster mirror.

```bash
$ docker pull docker.paddlepaddlehub.com/paddle:latest-dev-android
```

### Build the Inference Library

We can run the Docker image we just created to build the inference library of PaddlePaddle for Android using the command below:

```bash
$ docker run -it --rm -v $PWD:/paddle -w /paddle -e "ANDROID_ABI=armeabi-v7a" -e "ANDROID_API=21" paddle:dev-android ./paddle/scripts/paddle_build.sh build_android
```

The Docker image accepts two arguments `ANDROID_ABI` and `ANDROID_API`:

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

The ARM-64 architecture (`arm64-v8a`) requires at least level 21 of Android API.

The build command, [`paddle/scripts/paddle_build.sh build_android`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/scripts/paddle_build.sh) generates the [Android cross-compiling standalone toolchain](https://developer.android.com/ndk/guides/standalone_toolchain.html) based on the argument: `ANDROID_ABI` or `ANDROID_API`.  For information about other configuration arguments, please continue reading.

The above command generates and outputs the inference library in `$PWD/install_android` and puts third-party libraries in `$PWD/install_android/third_party`.

## Cross-Compiling on Linux

The Linux-base approach to cross-compile is to run steps in `Dockerfile.android` manually on a Linux x64 computer.

### Setup the Environment

To build for Android's, we need [Android NDK](
https://developer.android.com/ndk/downloads/index.html):

```bash
wget -q https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip
unzip -q android-ndk-r14b-linux-x86_64.zip
```

Android NDK includes everything we need to build the [*standalone toolchain*](https://developer.android.com/ndk/guides/standalone_toolchain.html), which in then used to build PaddlePaddle for Android.  (We plan to remove the intermediate stage of building the standalone toolchain in the near future.)

- To build the standalone toolchain for `armeabi-v7a` and Android API level 21:

```bash
your/path/to/android-ndk-r14b-linux-x86_64/build/tools/make-standalone-toolchain.sh \
        --arch=arm --platform=android-21 --install-dir=your/path/to/arm_standalone_toolchain
```
  
  The generated standalone toolchain will be in `your/path/to/arm_standalone_toolchain`.

- To build the standalone toolchain for `arm64-v8a` and Android API level 21:

```bash
your/path/to/android-ndk-r14b-linux-x86_64/build/tools/make-standalone-toolchain.sh \
        --arch=arm64 --platform=android-21 --install-dir=your/path/to/arm64_standalone_toolchain
```

  The generated standalone toolchain will be in `your/path/to/arm64_standalone_toolchain`.

### Cross-Compiling Arguments

CMake supports [choosing the toolchain](https://cmake.org/cmake/help/v3.0/manual/cmake-toolchains.7.html#cross-compiling).  PaddlePaddle provides [`android.cmake`](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/android.cmake), which configures the Android cross-compiling toolchain for CMake.  `android.cmake` is not required for CMake >= 3.7, which support Android cross-compiling. PaddlePaddle detects the CMake version, for those newer than 3.7, it uses [the official version](https://cmake.org/cmake/help/v3.7/manual/cmake-toolchains.7.html#cross-compiling).

Some other CMake arguments you need to know:

- `CMAKE_SYSTEM_NAME` must be `Android`.  This tells PaddlePaddle's CMake system to cross-compile third-party dependencies. This also changes some other CMake arguments like `WITH_GPU=OFF`, `WITH_AVX=OFF`, `WITH_PYTHON=OFF`, `WITH_RDMA=OFF`, `WITH_MKL=OFF` and `WITH_GOLANG=OFF`.
- `WITH_C_API` must be `ON`, to build the C-based inference library for Android.
- `WITH_SWIG_PY` must be `OFF` because the Android platform doesn't support SWIG-based API.

Some Android-specific arguments:

- `ANDROID_STANDALONE_TOOLCHAIN`: the absolute path of the Android standalone toolchain, or the path relative to the CMake build directory.  PaddlePaddle's CMake extensions would derive the cross-compiler, sysroot and Android API level from this argument.
- `ANDROID_TOOLCHAIN`: could be `gcc` or `clang`.  The default value is `clang`.
  - For CMake >= 3.7, it should anyway be `clang`.  For older versions, it could be `gcc`.
  - Android's official `clang` requires `glibc` >= 2.15.
- `ANDROID_ABI`: could be `armeabi-v7a` or `arm64-v8a`.  The default value is `armeabi-v7a`.
- `ANDROID_NATIVE_API_LEVEL`: could be derived from the value of `ANDROID_STANDALONE_TOOLCHAIN`.
- `ANROID_ARM_MODE`:
  - could be `ON` or `OFF`, and defaults to `ON`, when `ANDROID_ABI=armeabi-v7a`;
  - no need to specify when `ANDROID_ABI=arm64-v8a`.
- `ANDROID_ARM_NEON`: indicates if to use NEON instructions.
  - could be `ON` or `OFF`, and defaults to `ON`, when `ANDROID_ABI=armeabi-v7a`;
  - no need to specify when `ANDROID_ABI=arm64-v8a`.

Other useful arguments:

- `USE_EIGEN_FOR_BLAS`: indicates if using Eigen.  Could be `ON` or `OFF`, defaults to `OFF`.
- `HOST_C/CXX_COMPILER`: specifies the host compiler, which is used to build the host-specific protoc and target-specific OpenBLAS.  It defaults to the value of the environment variable `CC/C++`, or `cc/c++`.

Some frequent configurations for your reference:

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


There are some other arguments you might want to configure.

- `CMAKE_BUILD_TYPE=MinSizeRel` minimizes the size of library.
- `CMAKE_BUILD_TYPE-Release` optimizes the runtime performance.

Our own tip for performance optimization to use clang and Eigen or OpenBLAS:

- `CMAKE_BUILD_TYPE=Release`
- `ANDROID_TOOLCHAIN=clang`
- `USE_EIGEN_BLAS=ON` for `armeabi-v7a`, or `USE_EIGEN_FOR_BLAS=OFF` for `arm64-v8a`.

### Build and Install

After running `cmake`, we can run `make; make install` to build and install.

Before building, you might want to remove the `third_party` and `build` directories including pre-built libraries for other architectures.

After building，in the directory `CMAKE_INSTALL_PREFIX`, you will find three sub-directories:

- `include`: the header file of the inference library,
- `lib`: the inference library built for various Android ABIs,
- `third_party`: dependent third-party libraries built for Android.

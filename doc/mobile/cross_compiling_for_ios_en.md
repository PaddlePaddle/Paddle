# Build PaddlePaddle for iOS

This tutorial will walk you through cross compiling the PaddlePaddle library for iOS from the source in MacOS.

## Preparation

Apple provides Xcode for cross-compiling and IDE for iOS development. Download from App store or [here](https://developer.apple.com/cn/xcode/). To verify your installation, run command as follows

```bash
$ xcodebuild -version
Xcode 9.0
Build version 9A235
```

## Cross-compiling configurations

PaddlePaddle provides cross-compiling toolchain configuration documentation [cmake/cross_compiling/ios.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/ios.cmake), which has some default settings for frequently used compilers.

There are some mandatory environment variables need to be set before cross compiling PaddlePaddle for iOS:

- `CMAKE_SYSTEM_NAME`, CMake compiling target platform name, has to be `iOS`. PaddlePaddle CMake will compile all the third party dependencies and enforce some parameters (`WITH_C_API=ON`, `WITH_GPU=OFF`, `WITH_AVX=OFF`, `WITH_PYTHON=OFF`,`WITH_RDMA=OFF`) when this variable is set with value `iOS`.

- `WITH_C_API`, Whether to compile inference C-API library, has to be `ON`, since C-API is the only supported interface for inferencing in iOS.
- `WITH_SWIG_PY`, has to be `OFF`. It's not supported to inference or train via swig in iOS.

Optional environment variables for iOS are:

- `IOS_PLATFORM`, either `OS` (default) or `SIMULATOR`.
  - `OS`, build targets ARM-based physical devices like iPhone or iPad.
  - `SIMULATOR`, build targets x86 architecture simulators.
- `IOS_ARCH`, target architecture. By default, all architecture types will be compiled. If you need to specify the architecture to compile for, please find valid values for different `IOS_PLATFORM` settings from the table below:

    <table class="docutils">
    <colgroup>
      <col width="35%" />
      <col width="65%" />
    </colgroup>
    <thead valign="bottom">
      <tr class="row-odd">
      <th class="head">IOS_PLATFORM</th>
      <th class="head">IOS_ARCH</th>
    </tr>
    </thead>
    <tbody valign="top">
      <tr class="row-even">
      <td>OS</td>
      <td>armv7, armv7s, arm64 </td>
    </tr>
    <tr class="row-odd">
      <td>SIMULATOR</td>
      <td>i386, x86_64 </td>
    </tr>
    </tbody>
    </table>

- `IOS_DEPLOYMENT_TARGET`, minimum iOS version to deployment, `7.0` by default.
- `IOS_ENABLE_BITCODE`, whether to enable [Bitcode](https://developer.apple.com/library/content/documentation/IDEs/Conceptual/AppDistributionGuide/AppThinning/AppThinning.html#//apple_ref/doc/uid/TP40012582-CH35-SW3), values can be `ON/OFF`, `ON` by default.
- `IOS_USE_VECLIB_FOR_BLAS`, whether to use [vecLib](https://developer.apple.com/documentation/accelerate/veclib) framework for BLAS computing. values can be `ON/OFF`, `OFF` by default.
- `IOS_DEVELOPMENT_ROOT`, the path to `Developer` directory, can be explicitly set with your `/path/to/platform/Developer`. If left blank, PaddlePaddle will automatically pick the Xcode corresponding `platform`'s `Developer` directory based on your `IOS_PLATFORM` value.
- `IOS_SDK_ROOT`, the path to `SDK` root, can be explicitly set with your  `/path/to/platform/Developer/SDKs/SDK`. if left black, PaddlePaddle will pick the latest SDK in the directory of `IOS_DEVELOPMENT_ROOT`.

other settings：

- `USE_EIGEN_FOR_BLAS`, whether to use Eigen for matrix computing. effective when `IOS_USE_VECLIB_FOR_BLAS=OFF`. Values can be `ON/OFF`, `OFF` by default.
- `HOST_C/CXX_COMPILER`, host C/C++ compiler. Uses value from environment variable `CC/CXX` by default or `cc/c++` if `CC/CXX` doesn't exist.

some typical cmake configurations:

```bash
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DIOS_PLATFORM=OS \
      -DIOS_ARCH="armv7;arm64" \
      -DIOS_ENABLE_BITCODE=ON \
      -DIOS_USE_VECLIB_FOR_BLAS=ON \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_C_API=ON \
      -DWITH_TESTING=OFF \
      -DWITH_SWIG_PY=OFF \
      ..
```

```bash
cmake -DCMAKE_SYSTEM_NAME=iOS \
      -DIOS_PLATFORM=SIMULATOR \
      -DIOS_ARCH="x86_64" \
      -DIOS_USE_VECLIB_FOR_BLAS=ON \
      -DCMAKE_INSTALL_PREFIX=your/path/to/install \
      -DWITH_C_API=ON \
      -DWITH_TESTING=OFF \
      -DWITH_SWIG_PY=OFF \
      ..
```

You can set other compiling parameters for your own need. I.E. if you are trying to minimize the library size, set `CMAKE_BUILD_TYPE` with `MinSizeRel`; or if the performance is your concern, set `CMAKE_BUILD_TYPE` with `Release`. You can even manipulate the PaddlePaddle compiling procedure by manually set `CMAKE_C/CXX_FLAGS` values.

**TIPS for a better performance**:

- set `CMAKE_BUILD_TYPE` with `Release`
- set `IOS_USE_VECLIB_FOR_BLAS` with `ON`

## Build and install

After CMake, run following commands, PaddlePaddle will download the compile 3rd party dependencies, compile and install PaddlePaddle inference library.

```
$ make
$ make install
```

Please Note: if you compiled PaddlePaddle in the source directory for other platforms, do remove `third_party` and `build` directory within the source with `rm -rf` to ensure that all the 3rd party libraries dependencies and PaddlePaddle is newly compiled with current CMake configuration.

`your/path/to/install` directory will have following directories after `make install`:

- `include`, contains all the C-API header files.
- `lib`, contains PaddlePaddle C-API static library.
- `third_party` contains all the 3rd party libraries.

Please note: if PaddlePaddle library need to support both physical devices and simulators, you will need to compile correspondingly, then merge fat library with `lipo`.

Now you will have PaddlePaddle library compiled and installed, the fat library can be used in deep learning related iOS APPs. Please refer to C-API documentation for usage guides.

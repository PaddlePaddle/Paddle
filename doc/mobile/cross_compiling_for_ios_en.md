# Build PaddlePaddle for iOS

The cross-compiling of iOS requires a MacOS system.

## The Build Environment

Please download and install Xcode from App Store on the MacOS system.
After a successful installation, we should be able to run the
following command

```bash
xcodebuild -version
```

## Configure CMake Options

A complete list of CMake options related to iOS is
at
[here](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/ios.cmake).

Some of these options are mandatory:

- `CMAKE_SYSTEM_NAME`: The target platform; must be `iOS`.  This
  setting would automatically trigger some more changes, including
  `WITH_C_API=ON`, `WITH_GPU=OFF`, `WITH_AVX=OFF`, `WITH_PYTHON=OFF`,
  `WITH_RDMA=OFF`.
- `WITH_C_API`: If build the inference library; must be `ON`, because
  that is the only way we support inference on iOS.
- `WITH_SWIG_PY`: Must be `OFF` because we don't support that Python
  programs calls into C/C++ programs via SWIG on iOS.

Some optional configurations:

- `IOS_PLATFORM`: could be `OS` or `SIMULATOR`.  The default value is `OS`.
  - `OS`: the build result is supposed to run on an iPhone or an iPad
    with a real ARM CPU.
  - `SIMULATOR`: the build result can run on the x86 simulator
    provided by Xcode.
- `IOS_ARCH`: the value depends on that of `IOS_PLATFORM`:

  | `IOS_PLATFORM` | `IOS_ARCH` |
  |----------------|------------|
  | `OS`           | armv7, armv7s, arm64 |
  | `SIMULATOR`    | i386, x86_64 |

- `IOS_DEPLOYMENT_TARGET`: The supported minimum iOS version.  The
  default value is `7.0`.
- `IOS_ENABLE_BITCODE`: If
  use
  [Bitcode](https://developer.apple.com/library/content/documentation/IDEs/Conceptual/AppDistributionGuide/AppThinning/AppThinning.html#//apple_ref/doc/uid/TP40012582-CH35-SW3).
  The default value is `ON`.
- `IOS_USE_VECLIB_FOR_BLAS`: If
  use
  [vecLib](https://developer.apple.com/documentation/accelerate/veclib),
  a BLAS implementation for iOS.  The default value is `OFF`.
- `IOS_DEVELOPMENT_ROOT`: If not specified, PaddlePaddle's build
  system will set one according to the value of `IOS_PLATFORM`.
- `IOS_SDK_ROOT`: The iOS SDK directory.  If not specified,
  PaddlePaddle's build system will choose the most recent version in
  the directory of `IOS_DEVELOPMENT_ROOT`.

Other options:

- `USE_EIGEN_FOR_BLAS`: If use Eigen.  It is only reasonable to set
  this option when `IOS_USE_VECLIB_FOR_BLAS=OFF`.
- `HOST_C/CXX_COMPILER`: The host C/C++ compiler.  The default value
  is the value of the environment variable `CC/CXX`, or `cc` and `c++`
  respectively.


Here follows a commonly-used cmake configuration that builds binaries
for iPhone and iPad:

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


Here follows another configuration that builds binaries for the
simulator:

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

If we want to minimize the size of the generated binary files，we can
set `CMAKE_BUILD_TYPE=MinSizeRel`.  Or, if we want to optimize for
runtime performance, we can set `CMAKE_BUILD_TYPE=Release` and/or
`IOS_USE_VECLIB_FOR_BLAS=ON`.  We can also set `CMAKE_C/CXX_FLAGS` to
have more control of the building process.


## Build and Install

After the cmake command completes, run the following commands to
download third-party dependencies and build PaddlePaddle:

```
$ make
$ make install
```

Please be aware that if you had been built for other platforms, please
`rm -rf build` so to remove all existing intermediate build results.

After `make install`, you will find the following content in
`your/path/to/install`:

- `include`: the header files of the inference library,
- `lib`: the inference library,
- `third_party`: all built third-party libraries.

We can build a version that supports real iOS hardware and the
simulator, by building the IOS and SIMULATOR version separately, and
merge them using `lipo`.

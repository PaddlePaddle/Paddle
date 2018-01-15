# iOS平台编译指南
交叉编译iOS平台上适用的PaddlePaddle库，需要在MacOS系统上进行。本文的将介绍在MacOS上，从源码交叉编译iOS平台上适用的PaddlePaddle库。

## 准备交叉编译环境
Apple官方为iOS开发提供了完整的交叉编译工具和集成开发环境，用户从App Store下载安装Xcode即可。也可自行前往官网下载，[Xcode](https://developer.apple.com/cn/xcode/)。安装完成之后，可在命令行执行`xcodebuild -version`，判断是否安装成功。

```bash
$ xcodebuild -version
Xcode 9.0
Build version 9A235
```

## 配置交叉编译参数

PaddlePaddle为交叉编译提供了工具链配置文档[cmake/cross_compiling/ios.cmake](https://github.com/PaddlePaddle/Paddle/blob/develop/cmake/cross_compiling/ios.cmake)，以提供一些默认的编译器和编译参数配置。

交叉编译iOS版本的PaddlePaddle库时，有一些必须配置的参数：

- `CMAKE_SYSTEM_NAME`，CMake编译的目标平台，必须设置为`iOS`。在设置`CMAKE_SYSTEM_NAME=iOS`后，PaddlePaddle的CMake系统会自动编译所有的第三方依赖库，并且强制设置一些PaddlePaddle参数的值（`WITH_C_API=ON`、`WITH_GPU=OFF`、`WITH_AVX=OFF`、`WITH_PYTHON=OFF`、`WITH_RDMA=OFF`）。
- `WITH_C_API`，是否编译C-API预测库，必须设置为ON。在iOS平台上只支持使用C-API来预测。
- `WITH_SWIG_PY`，必须设置为`OFF`。在iOS平台上不支持通过swig调用来训练或者预测。

iOS平台可选配置参数：

- `IOS_PLATFORM`，可设置为`OS`（默认值）或`SIMULATOR`。
  - `OS`，构建目标为`arm`架构的iPhone或者iPad等物理设备。
  - `SIMULATOR`，构建目标为`x86`架构的模拟器平台。
- `IOS_ARCH`，目标架构。针对不同的`IOS_PLATFORM`，可设置的目标架构如下表所示，默认编译所有架构：

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

- `IOS_DEPLOYMENT_TARGET`，最小的iOS部署版本，默认值为`7.0`。
- `IOS_ENABLE_BITCODE`，是否使能[Bitcode](https://developer.apple.com/library/content/documentation/IDEs/Conceptual/AppDistributionGuide/AppThinning/AppThinning.html#//apple_ref/doc/uid/TP40012582-CH35-SW3)，可设置`ON/OFF`，默认值为`ON`。
- `IOS_USE_VECLIB_FOR_BLAS`，是否使用[vecLib](https://developer.apple.com/documentation/accelerate/veclib)框架进行BLAS矩阵计算，可设置`ON/OFF`，默认值为`OFF`。
- `IOS_DEVELOPMENT_ROOT`，`Developer`目录，可显式指定为`/path/to/platform/Developer`。若未显式指定，PaddlePaddle将会根据`IOS_PLATFORM`自动选择`Xcode`对应`platform`的`Developer`目录。
- `IOS_SDK_ROOT`，所使用`SDK`的根目录，可显式指定为`/path/to/platform/Developer/SDKs/SDK`。若未显式指定，PaddlePaddle将会自动选择`IOS_DEVELOPMENT_ROOT`目录下最新的`SDK`版本。

其他配置参数：

- `USE_EIGEN_FOR_BLAS`，是否使用Eigen库进行矩阵计算，在`IOS_USE_VECLIB_FOR_BLAS=OFF`时有效。可设置`ON/OFF`，默认值为`OFF`。
- `HOST_C/CXX_COMPILER`，宿主机的C/C++编译器。默认值为环境变量`CC/CXX`的值；若环境变量`CC/CXX`未设置，则使用`cc/c++`编译器。

常用的cmake配置如下：

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

用户还可根据自己的需求设置其他编译参数。比如希望最小化生成库的大小，可以设置`CMAKE_BUILD_TYPE`为`MinSizeRel`；若希望得到最快的执行速度，则可设置`CMAKE_BUILD_TYPE`为`Release`。亦可以通过手动设置`CMAKE_C/CXX_FLAGS`来影响PaddlePaddle的编译过程。

**性能TIPS**，为了达到最快的计算速度，在CMake参数配置上，有以下建议：

- 设置`CMAKE_BUILD_TYPE`为`Release`
- 设置`IOS_USE_VECLIB_FOR_BLAS=ON`，调用`vecLib`框架提供的BLAS函数进行矩阵计算。

## 编译和安装

CMake配置完成后，执行以下命令，PaddlePaddle将自动下载和编译所有第三方依赖库、编译和安装PaddlePaddle预测库。

```
$ make
$ make install
```

注意：如果你曾在源码目录下编译过其他平台的PaddlePaddle库，请先使用`rm -rf`命令删除`third_party`目录和`build`目录，以确保所有的第三方依赖库和PaddlePaddle代码都是针对新的CMake配置重新编译的。

执行完安装命令后，`your/path/to/install`目录中会包含以下内容：

- `include`目录，其中包含所有C-API的头文件
- `lib`目录，其中包含PaddlePaddle的C-API静态库
- `third_party`目录，其中包含所依赖的所有第三方库

注意，如果PaddlePaddle库需要同时支持真机和模拟器，则需要分别编译真机和模拟器版本，然后使用`lipo`工具合并fat库。

自此，PaddlePaddle库已经安装完成，用户可将合成的fat库用于深度学习相关的iOS App中，调用方法见C-API文档。

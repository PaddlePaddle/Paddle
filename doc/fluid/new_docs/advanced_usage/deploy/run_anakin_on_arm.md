## 源码编译 Anakin ##

目前Anakin支持ARM Android平台，采用Android NDK交叉编译工具链，已在mac os和centos上编译和测试通过。

### 安装概览 ###

* [系统需求](#0001)
* [安装第三方依赖](#0002)
* [Anakin源码编译](#0003)
* [验证安装](#0004)


### <span id = '0001'> 1. 系统需求 </span> ###

*  宿主机: linux, mac    
*  cmake 3.8.2+    
*  Android NDK r14, Linux 版本[从这里下载](https://dl.google.com/android/repository/android-ndk-r14b-linux-x86_64.zip)

### <span id = '0002'> 2. 安装第三方依赖 </span> ###

- 2.1 protobuf3.4.0     
   源码从这里[下载](https://github.com/google/protobuf/releases/tag/v3.4.0)    
 - 2.1.1 为宿主机编译protobuf     
 ```bash
   $ tar -xzf protobuf-3.4.0.tar.gz  
   $ cd protobuf-3.4.0   
   $ ./autogen.sh  
   $ ./configure    
   $ make  
   $ make check   
   $ make install
   ```
   上述 $make install 执行后，可在 /usr/local/include/google 找到 libprotobuf 所需的头文件,将整个google文件夹拷贝至Anakin/third-party/arm-android/protobuf/下，
   如有问题，请点[这里](https://github.com/google/protobuf/blob/v3.4.0/src/README.md)。
   然后将已经生成文件清除。
 ```bash
   $ make distclean
   ```
 - 2.1.1 交叉编译Android`armeabi-v7a`的protobuf，注意设置ANDROID_NDK的路径，以及ARCH_ABI、HOSTOSN的值，   
 ```bash

   $ export ANDROID_NDK=your_ndk_path 
   $ ARCH_ABI="arm-linux-androideabi-4.9"
   $ HOSTOSN="darwin-x86_64"
   $ export SYSROOT=$ANDROID_NDK/platforms/android-9/arch-arm  
   $ export PREBUILT=$ANDROID_NDK/toolchains/$ARCH_ABI
   $ export LDFLAGS="--sysroot=$SYSROOT"
   $ export LD="$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/arm-linux-androideabi/bin/ld $LDFLAGS"
   $ export LIBS="-llog $ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/libgnustl_static.a"
   $ export CPPFLAGS=""
   $ export INCLUDES="-I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/include/ -I$ANDROID_NDK/platforms/android-9/arch-arm/usr/include/ -I$ANDROID_NDK/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include/"
   $ export CXXFLAGS="-march=armv7-a -mfloat-abi=softfp -DGOOGLE_PROTOBUF_NO_RTTI --sysroot=$SYSROOT"
   $ export CCFLAGS="$CXXFLAGS"
   $ export CXX="$PREBUILT/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-g++ $CXXFLAGS"
   $ export CC="$CXX"
   $ export RANLIB="$ANDROID_NDK/toolchains/$ARCH_ABI/prebuilt/$HOSTOSN/bin/arm-linux-androideabi-ranlib"  
   $ ./autogen.sh  
   $ ./configure --host=arm-linux-androideabi --with-sysroot=$SYSROOT --enable-cross-compile --with-protoc=protoc --disable-shared CXX="$CXX" CC="$CC" LD="$LD"  
   $ make
  ```
  
  编译生成 *.a 静态库，若希望编译*.so 动态链接库 ，请在./configure参数中改--disable-shared为--disable-static --enable-shared。  
  生成文件在src/.libs/下，将生成的文件拷贝至Anakin/third-party/arm-android/protobuf/lib下。  
  在[cmake](../../cmake/find_modules.cmake)中更新`ARM_RPOTO_ROOT`的路径。        
  ```cmake
  set(ARM_RPOTO_ROOT "${CMAKE_SOURCE_DIR}/third-party/arm-android/protobuf")
  ```
  
- 2.2 opencv 2.4.3+(optional)    
    Anakin只在examples示例中使用opencv   
    Android系统的opencv从[这里下载](https://opencv.org/releases.html)    
    解压后将 `3rdparty/libs/armeabi-v7a`中的库文件拷贝到`libs/armeabi-v7a`    
    在[cmake](../../cmake/find_modules.cmake)中搜索`anakin_find_opencv`, 
    并设置 `include_directories` 和 `LINK_DIRECTORIES`为自己安装的库的路径。   
    ```cmake
    include_directories(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/jni/include/)
    LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third-party/arm-android/opencv/sdk/native/libs/armeabi-v7a/)
    ```
### <span id = '0003'> 3. Anakin源码编译 </span> ###

#### 编译Android版本

   克隆[源码](https://github.com/PaddlePaddle/Anakin/tree/arm)
```bash
    cd your_dir
    git clone https://github.com/PaddlePaddle/Anakin.git
    cd Anakin
    git fetch origin arm
    git checkout arm
  ```
  修改`android_build.sh`    
- 修改NDK路径    
  ```bash
    #modify "your_ndk_path" to your NDK path
    export ANDROID_NDK=your_ndk_path
  ```
- 修改ARM 处理器架构     
  对于32位ARM处理器, 将ANDROID_ABI 设置为 `armeabi-v7a with NEON`， 
  对于64位ARM处理器, 可以将ANDROID_ABI 设置为 `armeabi-v7a with NEON`或者`arm64-v8a`。        
  目前我们只支持 `armeabi-v7a with NEON`；`arm64-v8a` 还在开发中。      
  ```bash
      -DANDROID_ABI="armeabi-v7a with NEON"
  ```
- 设置Android API    
  根据Android系统的版本设置API level， 例如API Level 21 -> Android 5.0.1    
  ```bash
      -DANDROID_NATIVE_API_LEVEL=21
  ```

- 选择编译静态库或动态库    
  设置`BUILD_SHARED=NO`编译静态库    
  设置`BUILD_SHARED=YES`编译动态库    
  ```bash
      -DBUILD_SHARED=NO
  ```
- OpenMP多线程支持    
  设置`USE_OPENMP=YES`开启OpenMP多线程    
  ```bash
      -DUSE_OPENMP=YES
  ```
  
- 编译单测文件    
  设置`BUILD_WITH_UNIT_TEST=YES`将会编译单测文件    
    ```bash
        -DBUILD_WITH_UNIT_TEST=YES
    ```

- 编译示例文件    
  设置`BUILD_EXAMPLES=YES`将会编译示例文件    
    ```bash
        -DBUILD_EXAMPLES=YES
    ```
  
- 开启opencv    
  如果使用opencv，设置`USE_OPENCV=YES`    
    ```bash
        -DUSE_OPENCV=YES
    ```
    
- 开始编译    
  运行脚本 `android_build.sh` 将自动编译Anakin     
  ```bash
      ./android_build.sh
  ```

### <span id = '0004'> 4. 验证安装 </span> ###    
  编译好的库会放在目录`${Anakin_root}/output`下；    
  编译好的单测文件会放在`${Anakin_root}/output/unit_test`目录下；    
  编译好的示例文件会放在`${Anakin_root}/output/examples`目录下。
  
  对于Android系统，打开设备的调试模式，通过ADB可以访问的目录是`data/local/tmp`，通过ADB push将测试文件、模型和数据发送到设备目录， 运行测试文件。

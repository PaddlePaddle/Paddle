# 环境搭建
## 使用 docker
### 1. 安装 docker
安装 docker 的方式，参考官方文档 [https://docs.docker.com/install/](https://docs.docker.com/install/)
### 2. 使用 docker 搭建构建环境
首先进入 paddle-mobile 的目录下，执行 `docker build`
以 Linux/Mac 为例 (windows 建议在 'Docker Quickstart Terminal' 中执行)
```
$ docker build -t paddle-mobile:dev - < Dockerfile
```
使用 `docker images` 可以看到我们新建的 image
```
$ docker images
REPOSITORY      TAG     IMAGE ID       CREATED         SIZE
paddle-mobile   dev     33b146787711   45 hours ago    372MB
```
### 3. 使用 docker 构建
进入 paddle-mobile 目录，执行 docker run
```
$ docker run -it --mount type=bind,source=$PWD,target=/paddle-mobile paddle-mobile:dev
root@5affd29d4fc5:/ # cd /paddle-mobile
# 生成构建 android 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-android-neon.cmake
# 生成构建 linux 产出的 Makefile
root@5affd29d4fc5:/ # rm CMakeCache.txt
root@5affd29d4fc5:/ # cmake -DCMAKE_TOOLCHAIN_FILE=tools/toolchains/arm-linux-gnueabi.cmake
```
### 4. 设置编译选项
可以通过 ccmake 设置编译选项
```
root@5affd29d4fc5:/ # ccmake .
                                                     Page 1 of 1
 CMAKE_ASM_FLAGS
 CMAKE_ASM_FLAGS_DEBUG
 CMAKE_ASM_FLAGS_RELEASE
 CMAKE_BUILD_TYPE
 CMAKE_INSTALL_PREFIX             /usr/local
 CMAKE_TOOLCHAIN_FILE             /paddle-mobile/tools/toolchains/arm-android-neon.cmake
 CPU                              ON
 DEBUGING                         ON
 FPGA                             OFF
 LOG_PROFILE                      ON
 MALI_GPU                         OFF
 NET                              googlenet
 USE_EXCEPTION                    ON
 USE_OPENMP                       OFF
```
修改选项后，按 `c`, `g` 更新 Makefile
### 5. 构建
使用 make 命令进行构建
```
root@5affd29d4fc5:/ # make
```
### 6. 查看构建产出
构架产出可以在 host 机器上查看，在 paddle-mobile 的目录下，build 以及 test/build 下，可以使用 adb 指令或者 scp 传输到 device 上执行

## 不使用 docker
不使用 docker 的方法，可以直接用 cmake 生成 makefile 后构建。使用 ndk 构建 android 应用需要正确设置 NDK_ROOT。构建 linux 应用需要安装 arm-linux-gnueabi-gcc 或者类似的交叉编译工具，可能需要设置 CC，CXX 环境变量，或者在 tools/toolchains/ 中修改 arm-linux-gnueabi.cmake，或者增加自己需要的 toolchain file。

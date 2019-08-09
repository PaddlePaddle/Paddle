
# Paddle-lite-mobile开发指南

## 交叉编译

Paddle-lite-mobile 推荐在我们的Docker环境下交叉编译，减少环境配置上的不必要问题。

### 1. 拉取代码创建容器

```shell
$ git clone https://github.com/PaddlePaddle/Paddle.git
$ git checkout incubate/lite
```

编译docker环境:
`docker build --file paddle/fluid/lite/tools/Dockerfile.mobile --tag paddle-lite-mobile:latest . `

### 主要cmake选项
                
- `ARM_TARGET_OS` 代表目标操作系统， 目前支持 "android" "armlinux"， 默认是Android
- `ARM_TARGET_ARCH_ABI` 代表ARCH，支持输入"armv8"和"armv7"，针对OS不一样选择不一样。
    - `-DARM_TARGET_OS="android"` 时 
        - "armv8", 等效于 "arm64-v8a"。 default值为这个。
        - "armv7", 等效于 "armeabi-v7a"。 
    - `-DARM_TARGET_OS="armlinux"` 时 
        - "armv8", 等效于 "arm64"。 default值为这个。
        - "armv7hf", 等效于使用`eabihf`且`-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 `。
        - "armv7", 等效于使用`eabi`且`-march=armv7-a -mfloat-abi=softfp -mfpu=neon-vfpv4`。
- `ARM_TARGET_LANG` 代表目标编译的语言， 默认为gcc，支持 gcc和clang两种。

### 编译

基于`paddle-lite-mobile`镜像创建容器，并在容器内外建立目录映射关系：

```shell
$ docker run -it --name <yourname> --net=host --privileged -v <your-directory-path>:<your-directory-path> paddle-lite-mobile bash
```

参考build.sh下的 cmake arm编译需要的平台。

参考示例：

```shell
#!/bin/bash
cmake .. \
    -DWITH_GPU=OFF \
    -DWITH_LITE=ON \
    -DLITE_WITH_CUDA=OFF \
    -DLITE_WITH_X86=OFF \
    -DLITE_WITH_ARM=ON \
    -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
    -DWITH_TESTING=ON \
    -DWITH_MKL=OFF \
    -DARM_TARGET_OS="android" -DARM_TARGET_ARCH_ABI="arm64-v8a"

# fc层单测
make test_fc_compute_arm -j

```
### 在Android上执行

#### 1. 创建模拟器（如果使用真机则跳过此步骤）

```shell
# 创建Android avd (armv8)
$ echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
# 启动Android armv8 emulator
$ ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -verbose &

# 如果需要执行armv7版本，如下：
# $ echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
# $ ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -verbose &

# 退出所有模拟器
adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
```

#### 2. 上传编译文件到手机上

键盘上`crtl+q+p`同时摁下，切换到容器外（容器还在后台运行），将刚刚编译出的程序`adb push`到手机上。USB线连接手机，确保`adb devices`可以找到手机设备。
```shell
$ cd <paddle-repo>
$ adb push ./build/paddle/fluid/lite/kernels/arm/test_fc_compute_arm /data/local/tmp/

# 进入手机
$ adb shell # 若多台手机设备先用命令adb devices查看目标手机的序列码
$ cd /data/local/tmp

# 执行编译的程序
$ ./test_fc_compute_arm
```

### 在ARM LINUX下执行

拉取Linux arm64镜像
```shell
$ docker pull multiarch/ubuntu-core:arm64-bionic
```
运行容器并在内外建立目录映射关系
```shell
$ docker run -it --name <yourname> -v <your-directory-path>:<your-directory-path> multiarch/ubuntu-core:arm64-bionic
```
进入bin目录，并运行并文件
```shell
$ cd <bin-dir>
$ ./test_fc_compute_arm
```

# Q&A

#### 1. adb命令找不到：adb: command not found  
解决：`sudo apt install -y adb`   

#### 2. 明明手机USB连接电脑却显示找不到设备：`error: device not found`  
解决：
第一步`lsusb`命令查看插上拔下手机前后usb设备的变化情况，确定手机设备的ID。  假设`lsusb`命令执行显示`Bus 003 Device 011: ID 2717:9039  `，则ID是`0x2717`；  
第二步：创建`adb_usb.ini`文件并追加写入ID：`echo 0x2717 >> ~/.android/adb_usb.ini`；  
第三步：给手机添加权限`sudo vim /etc/udev/rules.d/70-android.rules`，根据第一步骤取得的`ATTRS{idVendor}`和`ATTRS{idProduct}`这两个属性值，在该文件加入该设备信息：
 `SUBSYSTEM=="usb", ATTRS{idVendor}=="2717", ATTRS{idProduct}=="9039",MODE="0666"`；  
第四步：重启USB服务：
```shell
$ sudo chmod a+rx /etc/udev/rules.d/70-android.rules
$ sudo service udev restart
```
第五步：重启adb服务，adb devices有设备说明adb安装成功。  
```shell
$ adb kill-server
$ sudo adb start-server
$ adb devices

# 若显示连接的手机设备，则表示成功
List of devices attached
5cb00b6 device
```
 

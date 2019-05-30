## Paddle-lite-mobile交叉编译指导

Paddle-lite-mobile开发环境目前推荐在Docker容器里，在容器里进行交叉编译安卓版本的Native C/C++代码，然后将可执行程序`adb push`到安卓手机上进行调试。
### 1. 拉取代码创建容器

```shell
$ git clone --recursive https://github.com/PaddlePaddle/Paddle.git
$ git checkout incubate/lite
```

先根据仓库下的`Dockerfile.android`文件生成对应的环境镜像。

```shell
$ cd <paddle-repo>
$ mkdir android-docker
$ cp Dockerfile.android ./android-docker/Dockerfile
$ cd android-docker
$ docker build -t paddle/paddle-lite-mobile .
```

完成后，可以看到：
```shell
$ docker images
REPOSITORY                        TAG            IMAGE ID            CREATED             SIZE
paddle/paddle-lite-mobile         latest       9c2000469891        5 hours ago         3.88GB
```

基于`paddle/paddle-lite`镜像创建容器，并在容器内外建立目录映射关系：

```shell
$ ddocker run -v <your-directory-path>:<your-directory-path> -tdi paddle/paddle-lite-mobile
# 启动成功会显示container_id
```

进入容器并切换到Paddle仓库目录：
```shell
$ docker exec -it <container_id> bash
$ cd <paddle-repo>
```

### 2. 交叉编译Paddle-lite-mobile的Native C/C++程序

创建名为`make_paddle_lite_mobile.sh`的文件：

```shell
$ touch make_paddle_lite_mobile.sh
$ chmod +x make_paddle_lite_mobile.sh
```

打开`make_paddle_lite_mobile.sh`文件然后将以下内容复制到该文件中，保存并退出：
```shell
#!/usr/bin/env bash

# build mobile
mkdir build
cd build

# cross-compile native cpp
cmake .. \
  -DWITH_GPU=OFF \
  -DWITH_MKL=OFF \
  -DWITH_LITE=ON \
  -DLITE_WITH_X86=OFF \
  -DLITE_WITH_ARM=ON \
  -DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
  -DLITE_WITH_CUDA=OFF \
  -DWITH_TESTING=ON

# fc层单测make test_fc_compute_arm -j
# 小模型单测#make cxx_api_lite_bin

```

### 3. 上传编译文件到手机上

键盘上`crtl+q+p`同时摁下，切换到容器外（容器还在后台运行），将刚刚编译出的程序`adb push`到手机上。USB线连接手机，确保`adb devices`可以找到手机设备。
```shell
$ cd <paddle-repo>
$ adb push ./build/paddle/fluid/lite/api/test_cxx_api_lite /data/local/tmp/

# 进入手机
$ adb shell # 若多台手机设备先用命令adb devices查看目标手机的序列码
$ cd /data/local/tmp

# 执行编译的程序$ ./test_cxx_api_lite
```

# runtime

TBD


### Q&A

#### 1. adb命令找不到：adb: command not found  
解决：`sudo apt install -y adb`   

#### 2. 明明手机USB连接电脑却显示找不到设备：`error: device not found`  
解决：第一步`lsusb`命令查看插上拔下手机前后usb设备的变化情况，确定手机设备的ID。  假设`lsusb`命令执行显示`Bus 003 Device 011: ID 2717:9039  `，则ID是`0x2717`；  
第二步：创建`adb_usb.ini`文件并追加写入ID：`echo 0x2717 >> ~/.android/adb_usb.ini`；  
第三步：给手机添加权限`sudo vim /etc/udev/rules.d/70-android.rules`，根据第一步骤取得的`ATTRS{idVendor}`和`ATTRS{idProduct}`这两个属性值，在该文件加入该设备信息： `SUBSYSTEM=="usb", ATTRS{idVendor}=="2717", ATTRS{idProduct}=="9039",MODE="0666"`；  
第四步：重启USB服务：```shell
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
#

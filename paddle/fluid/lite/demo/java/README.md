# Java Android Demo

要编译和跑起 ./android 文件夹下的 Android demo 程序 PaddlePredictor，你需要准备：

1. 一台能运行安卓程序的安卓手机
2. 一台带有AndroidStudio的开发机

## 手动编译

### 编译：
首先在PaddleLite的开发Docker镜像中，拉取最新PaddleLite代码，编译对应你手机架构的预测库，
下面我们以arm8 架构举例。进入paddlelite 目录，运行以下cmake 和make 命令：

```
mkdir -p build.lite.android.arm8.gcc
cd build.lite.android.arm8.gcc

cmake .. \
-DWITH_GPU=OFF \
-DWITH_MKL=OFF \
-DWITH_LITE=ON \
-DLITE_WITH_JAVA=ON \
-DLITE_WITH_CUDA=OFF \
-DLITE_WITH_X86=OFF \
-DLITE_WITH_ARM=ON \
-DLITE_WITH_LIGHT_WEIGHT_FRAMEWORK=ON \
-DWITH_TESTING=ON \
-DARM_TARGET_OS=android -DARM_TARGET_ARCH_ABI=armv8 -DARM_TARGET_LANG=gcc

make publish_inference_lite -j 4
```

Make完成后查看要存在`build.lite.android.arm8.gcc/paddle/fluid/lite/api/android/jni/libpaddle_lite_jni.so`
这个文件。该文件为PaddleLite c++ 动态链接库。接下来Android Java 代码会load 这个库来跑c++ 代码

### 把.so动态库拷贝进安卓demo程序：
把本文件夹下 demo/PaddlePredictor 载入到AndroidStudio。把上一步提到的`libpaddle_lite_jni.so`
拷贝进 `PaddlePredictor/app/src/main/jinLibs/所有架构文件夹下` 比如文件夹arm8里要包含该 .so文件：

### 把demo使用到的模型文件拷贝进安卓程序：
下载我们的5个模型文件，并复制到 `PaddlePredictor/app/src/main/assets` 这个文件夹中
需要拷贝的模型文件和下载地址：

    inception_v4_simple http://paddle-inference-dist.bj.bcebos.com/inception_v4_simple.tar.gz
    lite_naive_model    http://paddle-inference-dist.bj.bcebos.com/lite_naive_model.tar.gz
    mobilenet_v1        http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
    mobilenet_v2_relu   http://paddle-inference-dist.bj.bcebos.com/mobilenet_v2_relu.tar.gz
    resnet50            http://paddle-inference-dist.bj.bcebos.com/resnet50.tar.gz

下载完后，assets文件夹里要包含上面五个模型文件夹。

## 运行 Android 程序结果
以上准备工作完成，就可以开始Build ，安装，和跑安卓demo程序。当你运行PaddlePredictor 程序时，大概会等10秒，
然后看到类似以下字样：

    lite_naive_model output: 50.213173, -28.872887
    expected: 50.2132, -28.8729

    inception_v4_simple test:true
    time: 2078 ms

    resnet50 test:true
    time: 2078 ms

    mobilenet_v1 test:true
    time: 2078 ms

    mobilenet_v2 test:true
    time: 2078 ms

该 demo 程序跑我们的 5 个模型，第一个模型结果将真正的头两个数字输出，并在第二行附上期望的正确值。你应该要
看到他们的误差小于0.001。后面四个模型如果你看到 test:true 字样，说明模型输出通过了我们在 demo 程序里对其输出
的测试。time 代表该测试花费的时间。 

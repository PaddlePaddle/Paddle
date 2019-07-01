# C++ Android Demo
1. 使用`paddle/fluid/lite/tools/Dockerfile.mobile`生成docker镜像
2. 运行并进入docker镜像环境，执行`wget http://http://paddle-inference-dist.bj.bcebos.com/inference_lite_lib.android.armv8.tar.gz `下载所需demo环境。(armv7 demo可使用命令`wget http://http://paddle-inference-dist.bj.bcebos.com/inference_lite_lib.android.armv7.tar.gz` 进行下载)。
3. 解压下载文件`tar zxvf inference_lite_lib.android.armv8.tar.gz `
4. 执行以下命令准备模拟器环境
```shell
# armv8
adb kill-server
adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
echo n | avdmanager create avd -f -n paddle-armv8 -k "system-images;android-24;google_apis;arm64-v8a"
echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv8 -noaudio -no-window -gpu off -port 5554 &
sleep 1m
```
```shell
# armv7
adb kill-server
adb devices | grep emulator | cut -f1 | while read line; do adb -s $line emu kill; done
echo n | avdmanager create avd -f -n paddle-armv7 -k "system-images;android-24;google_apis;armeabi-v7a"
echo -ne '\n' | ${ANDROID_HOME}/emulator/emulator -avd paddle-armv7 -noaudio -no-window -gpu off -port 5554 &
sleep 1m
```
5. 准备模型、编译并运行完整api的demo
```shell
cd inference_lite_lib.android.armv8/demo/cxx/mobile_full
wget http://http://paddle-inference-dist.bj.bcebos.com/mobilenet_v1.tar.gz
tar zxvf mobilenet_v1.tar.gz
make
adb -s emulator-5554 push mobilenet_v1 /data/local/tmp/
adb -s emulator-5554 push mobilenetv1_full_api /data/local/tmp/
adb -s emulator-5554 shell chmod +x /data/local/tmp/mobilenetv1_full_api
adb -s emulator-5554 shell "/data/local/tmp/mobilenetv1_full_api --model_dir=/data/local/tmp/mobilenet_v1 --optimized_model_dir=/data/local/tmp/mobilenet_v1.opt"
```
运行成功将在控制台输出预测结果的前10个类别的预测概率

6. 编译并运行轻量级api的demo
```shell
cd ../mobile_light
make
adb -s emulator-5554 push mobilenetv1_light_api /data/local/tmp/
adb -s emulator-5554 shell chmod +x /data/local/tmp/mobilenetv1_light_api
adb -s emulator-5554 shell "/data/local/tmp/mobilenetv1_light_api --model_dir=/data/local/tmp/mobilenet_v1.opt
```
运行成功将在控制台输出预测结果的前10个类别的预�

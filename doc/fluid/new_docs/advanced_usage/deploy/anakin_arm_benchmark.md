# Anakin ARM 性能测试

## 测试环境和参数:
+ 测试模型Mobilenetv1, mobilenetv2, mobilenet-ssd
+ 采用android ndk交叉编译，gcc 4.9，enable neon， ABI： armveabi-v7a with neon -mfloat-abi=softfp
+ 测试平台
   - 荣耀v9(root): 处理器:麒麟960, 4 big cores in 2.36GHz, 4 little cores in 1.8GHz
   - nubia z17:处理器:高通835, 4 big cores in 2.36GHz, 4 little cores in 1.9GHz
   - 360 N5:处理器:高通653, 4 big cores in 1.8GHz, 4 little cores in 1.4GHz
+ 多线程：openmp
+ 时间：warmup10次，运行10次取均值
+ ncnn版本：来源于github的master branch中commits ID：307a77f04be29875f40d337cfff6df747df09de6（msg:convert            LogisticRegressionOutput)版本
+ TFlite版本：来源于github的master branch中commits ID：65c05bc2ac19f51f7027e66350bc71652662125c（msg:Removed unneeded file copy that was causing failure in Pi builds)版本

在BenchMark中本文将使用**`ncnn`**、**`TFlite`**和**`Anakin`**进行性能对比分析

## BenchMark model

> 注意在性能测试之前，请先将测试model通过[External Converter](#10003)转换为Anakin model
> 对这些model，本文在ARM上进行多线程的单batch size测试。

- [Mobilenet v1](#11)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [Mobilenet v2](#22)  *caffe model 可以在[这儿](https://github.com/shicai/MobileNet-Caffe)下载*
- [mobilenet-ssd](#33)  *caffe model 可以在[这儿](https://github.com/chuanqi305/MobileNet-SSD)下载*

### <span id = '11'> mobilenetv1 </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|107.7ms|61.1ms|38.2ms|152.8ms|85.2ms|51.9ms|152.6ms|nan|nan|
   |高通835|105.7ms|63.1ms|~~46.8ms~~|152.7ms|87.0ms|~~92.7ms~~|146.9ms|nan|nan|
   |高通653|120.3ms|64.2ms|46.6ms|202.5ms|117.6ms|84.8ms|158.6ms|nan|nan| 

### <span id = '22'> mobilenetv2 </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|93.1ms|53.9ms|34.8ms|144.4ms|84.3ms|55.3ms|100.6ms|nan|nan|
   |高通835|93.0ms|55.6ms|41.1ms|139.1ms|88.4ms|58.1ms|95.2ms|nan|nan|
   |高通653|106.6ms|64.2ms|48.0ms|199.9ms|125.1ms|98.9ms|108.5ms|nan|nan|

### <span id = '33'> mobilenet-ssd </span>

   |platform | Anakin (1) | Anakin (2) | Anakin (4) | ncnn (1) | ncnn (2) | ncnn (4) | TFlite (1) | TFlite (2) | TFlite (4)| 
   |:---: | :---: | :---: | :---:| :---:| :---:| :---:| :---:| :---:| :---:|
   |麒麟960|213.9ms|120.5ms|74.5ms|307.9ms|166.5ms|104.2ms|nan|nan|nan|
   |高通835|213.0ms|125.7ms|~~98.4ms~~|292.9ms|177.9ms|~~167.8ms~~|nan|nan|nan|
   |高通653|236.0ms|129.6ms|96.0ms|377.7ms|228.9ms|165.0ms|nan|nan|nan

## How to run those Benchmark models?

1. 首先, 使用[External Converter](../docs/Manual/Converter_en.md)对caffe model 进行转换
2. 然后将转换后的Anakin model和编译好的benchmark_arm 二进制文件通过'adb push'命令上传至测试机
3. 接着在测试机含有Anakin model的目录中运行'./benchmark_arm ./ anakin_model.anakin.bin 1 10 10 1' 命令
4. 最后，终端显示器上将会打印该模型的运行时间
5. 其中运行命令的参数个数和含义可以通过运行'./benchmark_arm'看到

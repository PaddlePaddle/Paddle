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

<p align="center">
    <table>
        <thead>
            <tr>
                <th>platform</th>
                <th>Anakin (1)</th>
                <th>Anakin (2)</th>
                <th>Anakin (4)</th>
                <th>ncnn (1)</th>
                <th>ncnn (2)</th>
                <th>ncnn (4)</th>
                <th>TFlite (1)</th>
                <th>TFlite (2)</th>
                <th>TFlite (4)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>麒麟960</td>
                <td>107.7ms</td>
                <td>61.1ms</td>
                <td>38.2ms</td>
                <td>152.8ms</td>
                <td>85.2ms</td>
                <td>51.9ms</td>
                <td>152.6ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通835</td>
                <td>105.7ms</td>
                <td>63.1ms</td>
                <td>
                    <del>46.8ms</del>
                </td>
                <td>152.7ms</td>
                <td>87.0ms</td>
                <td>
                    <del>92.7ms</del>
                </td>
                <td>146.9ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通653</td>
                <td>120.3ms</td>
                <td>64.2ms</td>
                <td>46.6ms</td>
                <td>202.5ms</td>
                <td>117.6ms</td>
                <td>84.8ms</td>
                <td>158.6ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
        </tbody>
    </table>
</p>


### <span id = '22'> mobilenetv2 </span>

<p align="center">
    <table>
        <thead>
            <tr>
                <th>platform</th>
                <th>Anakin (1)</th>
                <th>Anakin (2)</th>
                <th>Anakin (4)</th>
                <th>ncnn (1)</th>
                <th>ncnn (2)</th>
                <th>ncnn (4)</th>
                <th>TFlite (1)</th>
                <th>TFlite (2)</th>
                <th>TFlite (4)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>麒麟960</td>
                <td>93.1ms</td>
                <td>53.9ms</td>
                <td>34.8ms</td>
                <td>144.4ms</td>
                <td>84.3ms</td>
                <td>55.3ms</td>
                <td>100.6ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通835</td>
                <td>93.0ms</td>
                <td>55.6ms</td>
                <td>41.1ms</td>
                <td>139.1ms</td>
                <td>88.4ms</td>
                <td>58.1ms</td>
                <td>95.2ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通653</td>
                <td>106.6ms</td>
                <td>64.2ms</td>
                <td>48.0ms</td>
                <td>199.9ms</td>
                <td>125.1ms</td>
                <td>98.9ms</td>
                <td>108.5ms</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
        </tbody>
    </table>
</p>

### <span id = '33'> mobilenet-ssd </span>

<p align="center">
    <table>
        <thead>
            <tr>
                <th>platform</th>
                <th>Anakin (1)</th>
                <th>Anakin (2)</th>
                <th>Anakin (4)</th>
                <th>ncnn (1)</th>
                <th>ncnn (2)</th>
                <th>ncnn (4)</th>
                <th>TFlite (1)</th>
                <th>TFlite (2)</th>
                <th>TFlite (4)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>麒麟960</td>
                <td>213.9ms</td>
                <td>120.5ms</td>
                <td>74.5ms</td>
                <td>307.9ms</td>
                <td>166.5ms</td>
                <td>104.2ms</td>
                <td>nan</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通835</td>
                <td>213.0ms</td>
                <td>125.7ms</td>
                <td>
                    <del>98.4ms</del>
                </td>
                <td>292.9ms</td>
                <td>177.9ms</td>
                <td>
                    <del>167.8ms</del>
                </td>
                <td>nan</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
            <tr>
                <td>高通653</td>
                <td>236.0ms</td>
                <td>129.6ms</td>
                <td>96.0ms</td>
                <td>377.7ms</td>
                <td>228.9ms</td>
                <td>165.0ms</td>
                <td>nan</td>
                <td>nan</td>
                <td>nan</td>
            </tr>
        </tbody>
    </table>
</p>

## How to run those Benchmark models?

1. 首先, 使用[External Converter](../docs/Manual/Converter_en.md)对caffe model 进行转换
2. 然后将转换后的Anakin model和编译好的benchmark_arm 二进制文件通过'adb push'命令上传至测试机
3. 接着在测试机含有Anakin model的目录中运行'./benchmark_arm ./ anakin_model.anakin.bin 1 10 10 1' 命令
4. 最后，终端显示器上将会打印该模型的运行时间
5. 其中运行命令的参数个数和含义可以通过运行'./benchmark_arm'看到

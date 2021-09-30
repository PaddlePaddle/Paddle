# 在 IPU 上识别手写数字 

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/02.recognize_digits/image/mnist_example_image.png?raw=true" width="400"><br/>
图片 1. MNIST 图片样例
</p>

## 在 `IPU` 上训练精度对齐

这里我们采用由5个构建模块组成的 `LeNet-5` 模型。虽然这些模型在现代框架卷积块中会被融合为一个大算子。

在 `CPU` 上训练 MNIST 数据集可以通过4次迭代达到 `98.3x%` 的准确率， 而在 `IPU` 上收敛速度，更快一些，同样的数据集划分和迭代次数，可以达到 `99.9y` 的准确度。

```bash
Epoch 4, batch 100, Cost 0.054509, Validation Cost 0.000850
Epoch 4, batch 200, Cost 0.006553, Validation Cost 0.000850
Epoch 4, batch 300, Cost 0.012887, Validation Cost 0.000850
Epoch 4, batch 400, Cost 0.005729, Validation Cost 0.000850
Epoch 4, batch 500, Cost 0.011038, Validation Cost 0.000850
Epoch 4, batch 600, Cost 0.031277, Validation Cost 0.000850
Epoch 4, batch 700, Cost 0.032082, Validation Cost 0.000850
Epoch 4, batch 800, Cost 0.054537, Validation Cost 0.000850
Epoch 4, batch 900, Cost 0.026884, Validation Cost 0.000850
Best pass is 4, validation average loss is 0.000850238038765383
The classification accuracy is 99.98%
```
<p align="center">
 在 IPU 上训练 MNIST
</p>

## 在 IPU 上训练

> bash train_with_ipu.sh

#### 可选项
The program has a few comand line options.

程序有一些命令后选项

For MNIST dataset, it is enough to train on a single IPU, it is enough for precision alignmnet. We will support Model replica and data parallel for feature test purpose soon.

MNSIT 数据集足够小，可以在1个IPU上运行。对于精度对齐是足够的。我们后面会支持数据并行用于特性测试。

`-h`                   展示信息.  

`--use_ipu`            在 IPU 上训练.

`--num_ipus`           IPU 设备数目，默认1.

`--no_pipelining`      如果计算图没有被分到多个设备上，不同IPU的模型不会执行"pipelining"并发策略.

`--replication-factor` 模型拷贝数目，必须是IPU整数倍.

例子:

    python train.py \
    --use_ipu True \
    --num_ipus 1 \
    --no_pipelining 

## 通过 Python 前端在 `IPU` 做推理

> bash infer_with_ipu.sh

示例：

```
(py37_paddle-ipu) [docker-λ>] leiw@gbnwx-pod006-3-in_docker_dev:~/Paddle/python/paddle/fluid/tests/unittests/ipu/test_dataset/mnist$ bash infer_with_ipu.sh
[09/18 07:14:40] mnist:infer INFO: Reading data ...
[09/18 07:14:40] mnist:infer INFO: Complete reading image infer_3.png
[09/18 07:14:40] mnist:infer INFO: Constructing the computation graph ...
[09/18 07:15:12] mnist:infer INFO: Computation graph built.
[09/18 07:15:12] mnist:infer INFO: Change batch size of var %s from %d to %d
[09/18 07:15:12] mnist:infer INFO: Drawing IR graph ...
[09/18 07:15:12] mnist:infer INFO: Complete drawing.
digit hand write number picture is recognized as : 3
```
```

## 通过 Analysis API (c++) 在 `IPU` 上做推理

我们将加入相关示例


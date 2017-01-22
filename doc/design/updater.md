# Design Doc: New Paddle API For Updater

In gradient-base optimization algorithms, the parameters are updated using the gradients in each iteration. We call the component that do update work Updater.

The main method of an Updater is update(parameters), there may have more then one parameters is multi-layer neural network, the Updater cann
update each parameter one by one with updater.update(parameter)

```python
gm = GradientMachine()
updater = paddle.Updater(optimization_method=SgdOptimizer, learning_rate=0.1, ...)
for data_batch in data.read_data():
    gm.foward(data_batch)
    gm.backward()
    updater.update(gm)
```

在之前的工作 #1108 中，开始进行python api v2的开发工作，抽象出来了 layers，optimizer，data等组件。

之前的重构是对原有的[config_parser.py](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/trainer/config_parser.py) 做了一层封装，对于这种封装的改造方式，王益老师的意见是：

>之前我们讨论说先用目前的“不需要定义多个.py文件“的API把所有demo都重写一遍，然后再来看应该如何完善API。
但是我刚才又想了一下，是不是API里如果有一些明显的问题，可以先修正问题，然后再来重写mnist之后的下一个demo。这样效率更高？
在写mnist的时候，我们不要 import * from 已有的Python packages，而是 `copy-n-paste` 已有的package 到 paddle.v2。这样我们就可以在”把mnist demo写得顾名思义“这个过程里，修改copy 过来的实现。当我们针对每个demo重复这个过程之后，我们是不是就得到了一个完备的v2 API了。

按着这种思路，在上述的几个组件中感觉optimizer这个组件相对比较独立，所以决定第一步先把updater和相关配置独立出来。

主要方案：

* 1，对外接口方面，在[optimizer](https://github.com/PaddlePaddle/Paddle/pull/1108/commits/2b988b47768b017abf08e49298d72c17c8bf89ad)的基础上继续完善。
* 2，配置生成的方案，主要需要重构 [settings](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/trainer_config_helpers/optimizers.py#L358) 以及涉及到的 [config_parser](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/trainer/config_parser.py#L3272)中的部分内容，主要需要做的是将这部分代码及相关代码，从config_parser中单独抽离出来，放到v2下，并且改变之前通过回调的生成配置的方式，直接生成对应的proto。

这样做的好处：

* 1，optimizer相对独立，且功能没有layer配置那么复杂，比较容易着手。
* 2，optimizer的部分从config_parser中独立出来之后，会简化后面重新定义layer和network部分的工作。

使用方式：
```python
    updater = paddle.v2.Updater(
        learning_method=paddle.v2.optimizer.AdamOptimizer(),
        learning_rate=1e-4,
        model_average=paddle.v2.optimizer.ModelAverage(average_window=0.5),
        regularization=paddle.v2.optimizer.L2Regularization(rate=0.5))
```

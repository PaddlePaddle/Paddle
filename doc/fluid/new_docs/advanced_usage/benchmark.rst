#################
如何进行基准测试
#################

本文介绍如何给深度学习框架做基准测试。基准测试主要包含验证模型的精度和性能两方面，下文包含搭建测试环境，选择基准测试模型，验证测试结果等几方面内容。

验证深度学习框架，可分为训练和测试两个阶段， 验证指标略有不同，本文只介绍训练阶段的指标验证。训练阶段关注的是模型训练集上的精度，训练集是完备的，因此关注大batch\_size下的训练速度,关注吞吐量，例如图像模型常用的batch\_size=128, 多卡情况下会加大；预测阶段关注的是在测试集上的精度，线上服务测试数据不能提前收集，因此关注小batch\_size下的预测速度，关注延迟，例如预测服务常用的batch\_size=1, 4等。

`Fluid <https://github.com/PaddlePaddle/Paddle>`__ 是PaddlePaddle从0.11.0版本开始引入的设计，本文的基准测试在该版本上完成。


环境搭建
""""""""""""

基准测试中模型精度和硬件、框架无关，由模型结构和数据共同决定；性能方面由测试硬件和框架性能决定。框架基准测试为了对比框架之间的差异，控制硬件环境，系统库等版本一致。下文中的对比实验都在相同的硬件条件和系统环境条件下进行.


不同架构的GPU卡性能差异巨大，在验证模型在GPU上训练性能时，可使用NVIDIA提供的工具:code `nvidia-smi` 检验当前使用的GPU型号，如果测试多卡训练性能，需确认硬件连接是 `nvlink <https://zh.wikipedia.org/zh/NVLink>`__ 或 `PCIe <https://zh.wikipedia.org/zh-hans/PCI_Express>`__ 。 同样地，CPU型号会极大影响模型在CPU上的训练性能。可读取`/proc/cpuinfo`中的参数，确认当前正在使用的CPU型号。

下载GPU对应的Cuda Tool Kit和 Cudnn，或者使用NVIDIA官方发布的nvidia-docker镜像 `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`__, 镜像内包含了Cuda和Cudnn，本文采用这种方式。 Cuda Tool Kit包含了GPU代码使用到的基础库，影响在此基础上编译出的Fluid二进制运行性能。

准备好Cuda环境后，从github上的下载Paddle并源码编译，会生成对应的最适合当前GPU的sm\_arch二进制\ `sm\_arch <https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html>`__\ 。另外，cudnn对卷积类任务影响巨大，在基准测试中需要小版本一致，例如Cudnn7.0.2与Cudnn7.1.4在Resnet上有5%以上差异。


选择基准模型
""""""""""""

对框架做基准测试，需要覆盖不同训练任务和不同大小的模型，本文中选取了图像和NLP的最为常用的5个模型。

============  ============  =================  ============
任务种类        模型名称       网络结构         数据集     
============  ============  =================  ============
图像分类      mnist         Lenet              mnist
图像分类      VGG           VGG-16             Flowers102
图像分类      Resnet        Resnet-50          Flowers102
文本分类      Stacked-LSTM  Stacked-LSTM       IMDB 
机器翻译      seq-seq       Stacked-LSTM       wmt14 
============  ============  =================  ============

其中mnist, VGG, Resnet属于CNN模型, stacked-lstm, seq2seq代表RNN模型。
`benchmark <https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid>`__
基准模型测试脚本中，均跳过了前几个batch的训练过程，原因是加载数据和分配显存受系统当前运行情况影响，会导致统计性能不准确。运行完若干个轮次后，统计对应指标。


基准模型的数据的选择方面，数据量大且验证效果多的公开数据集为首选。图像模型VGG和resnet, 本文选择了 `flowers102 <http://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`__ ，图像大小预处理为和Imagenet相同大小，因此性能可直接对比
NLP模型的公开且影响力大数据集较少，seq2seq模型选择了wmt14数据，stacked-lstm模型中选择了 `imdb <https://www.imdb.com/interfaces/>`__ 数据。


注意，图像模型每条样本大小相同，图像经过变换后大小一致，因此经过的计算路径基本相同，计算速度和显存占用波动较小，可以从若干个batch的数据中采样得到当前的训练性能数据。而NLP模型由于样本长度不定，计算路径和显存占用也不相同，因此只能完整运行若干个轮次后，统计速度和显存消耗。
显存分配是特别耗时的操作，因此Fluid默认会占用所有可用显存空间形成显存池，用以加速计算过程中的显存分配。如果需要统计模型真实显存消耗，可设置环境变量`FLAGS_fraction_of_gpu_memory_to_use=0.0`，观察最大显存开销。


测试过程
""""""""""""

-  CPU 单机单线程测试

测试CPU上单线程的性能，先设置CUDA的环境变量为空，``CUDA_VISIBLE_DEVICES=``，并通过环境变量关闭OpenMP和MKL的多线程 ``OMP_NUM_THREADS=1``， ``MKL_NUM_THREADS=1;``。
然后代码中设置为使用CPUPlace，如果使用Paddle代码库中的脚本，只需要命令行参数传入 use_gpu=False即可。

.. code-block:: python

    >>> import paddle.fluid as fluid
    >>> place = fluid.CPUPlace() 

.. code:: bash

    docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark paddlepaddle/paddle:latest-dev /bin/bash


-  GPU 单机单卡测试

本教程使用了Cuda8, Cudnn7.0.1。来源为:code `nvidia/cuda:8.0-cudnn7-devel-ubuntu16.04`

.. code:: bash

    nvidia-docker run -it --name CASE_NAME --security-opt seccomp=unconfined -v $PWD/benchmark:/benchmark -v /usr/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu paddlepaddle/paddle:latest-dev /bin/bash
在单卡上测试，设置CUDA的环境变量使用一块GPU，``CUDA_VISIBLE_DEVICES=0``
然后代码中设置为使用CUDAPlace，如果使用Paddle代码库中的脚本，只需要命令行参数传入 use_gpu=True即可。

.. code-block:: python

    >>> import paddle.fluid as fluid
    >>> place = fluid.CUDAPlace(0) // 0 指第0块GPU


测试结果
""""""""""""

本教程对比相同环境下的Fluid0.12.0和TensorFlow1.4.0的性能表现。
硬件环境为 CPU: Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHz, GPU: TITAN X(Pascal) 12G x 1, Nvidia-Driver 384.90。
系统环境为Ubuntu 16.04.3 LTS, 本文中采用了docker环境，系统版本为nvidia-docker17.05.0-ce。
测试的Fluid版本为\ `v.0.12.0 <https://github.com/PaddlePaddle/Paddle/releases/tag/v.0.12.0>`__ 。
TensorFlow版本为\ `v.1.4.0-rc1 <https://github.com/tensorflow/tensorflow/tree/v1.4.0-rc1>`__ 。
使用的脚本和配置见\ `benchmark <https://github.com/PaddlePaddle/Paddle/tree/develop/benchmark/fluid>`__ 。
图表中统计单位为samples/秒。

- CPU 单机单线程测试结果

  ================  ====================  ===================
   Speed            Fluid CPU              TensorFlow CPU    
  ================  ====================  ===================
  mnist             1298.75 samples/s     637.57 samples/s  
  VGG-16            0.4147 images/s       0.1229 images/s   
  Resnet-50         1.6935 images/s       0.3657 images/s   
  Stacked-LSTM      472.3225 words/s      48.2293words/s    
  Seq2Seq           217.1655 words/s      28.6164 words/s   
  ================  ====================  ===================

- GPU 单机单卡测试结果

  =============== =====================  =================
   Speed           Fluid GPU              TensorFlow GPU      
  =============== =====================  =================
   mnist           19710.90 samples/s    15576.3 samples/s        
   VGG-16          59.83327 images/s     40.9967 images/s    
   Resnet-50       105.84412             97.8923 images/s    
   Stacked-LSTM    1319.99315            1608.2526 words/s   
   Seq2Seq         7147.89081            6845.1161 words/s   
  =============== =====================  =================

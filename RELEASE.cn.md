# v0.11.0版本

## PaddlePaddle Fluid

- PaddlePaddle发布版本v0.11.0包含一个新的特性*PaddlePaddle Fluid*. Fluid 是设计用来让用户像Pytorch和Tensorflow Eager Execution一样执行程序。在这些系统中，不再有*模型*这个概念，应用也不再包含一个用于描述Operator图或者一系列层的符号描述，而是像通用程序那样描述训练或者预测的过程。而Fluid与PyTorch或Eager Execution的区别在于Fluid不依赖Python提供的控制流，例如 if-else-then或者for，而是提供了基于C++实现的控制流并暴露了对应的用with语法实现的Python接口。例如：

  https://github.com/PaddlePaddle/Paddle/blob/3df78ed2a98d37f7ae6725894cc7514effd5664b/python/paddle/v2/fluid/tests/test_while_op.py#L36-L44

- 在v0.11.0版本中，我们提供了一个C++类`Executor`用于运行一个Fluid程序。Executor类似一个解释器。在未来的版本中，我们将提升和优化Executor成为一个调试器，就像GDB。并可能提供一些编译器，这个编译器会读取一个上文所描述的应用然后编译成一个等价的
源代码，这个源代码可以被nvcc编译成可以使用CUDA的二进制，或者被icc编译成可以充分利用Intel CPU的二进制。


## 新特点

* 发布 `PaddlePaddle Fluid`。
* 增加了用于模型预测的C-API。
* 用Fluid API实现了一个简单的GAN的例子。
* 增加了关于性能调优的文档。
* 为`paddle.v2.dataset`下载数据集提供了重试机制.
* C++中使用protobuf-lite替换protobuf减少了二进制的大小。
* 发布了新特性 [Elastic Deep Learning (EDL)](https://github.com/PaddlePaddle/cloud/tree/develop/doc/autoscale/experiment).
* 基于Bazel API利用cmake实现了一个的新的构建系统函数库。
* 当使用编译选项`WITH_MKL=ON`时自动下载和编译Intel® [MKLML](https://github.com/01org/mkl-dnn/releases/download/v0.11/mklml_lnx_2018.0.1.20171007.tgz) 函数库.
* [Intel® MKL-DNN on PaddlePaddle](https://github.com/PaddlePaddle/Paddle/tree/develop/doc/design/mkldnn):
  - 完成了 11个 MKL-DNN 层: Convolution, Fully connectivity, Pooling, ReLU, Tanh, ELU, Softmax, BatchNorm, AddTo, Concat, LRN。
  - 完成了 3个 MKL-DNN 网络: VGG-19, ResNet-50, GoogleNet
  - 基于Intel Skylake 6148 CPU的[性能测试](https://github.com/PaddlePaddle/Paddle/blob/develop/benchmark/IntelOptimizedPaddle.md) : 相对于MKLML有2~3倍的训练加速。
* 增加 [softsign activation](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/activation.html#softsign)
* 增加 [dot product layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#dot-prod)
* 增加 [L2 distance layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#l2-distance)
* 增加 [sub-nested sequence layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#sub-nested-seq)
* 增加 [kmax sequence score layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#kmax-sequence-score)
* 增加 [sequence slice layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#seq-slice)
* 增加 [row convolution layer](http://www.paddlepaddle.org/docs/develop/documentation/zh/api/v2/config/layer.html#row-conv)
* 增加移动端友好的网页

## 改进

* 使用一个Python`whl`包即可安装.
* [V2 API可以实现用户定制化评估](https://github.com/PaddlePaddle/models/tree/develop/ltr#训练过程中输出自定义评估指标)。
* 将 `PADDLE_ONLY_CPU` 改为 `PADDLE_WITH_GPU`, 因为我们会支持多种设备。
* 删除了有一些bug的BarrierStat。
* 清理和删除了paddle::Parameter中未使用的函数。
* 删除了ProtoDataProvider。
* Huber loss同时支持回归和分类。
* 为sequence pooling 层增加`stride`参数。
* v2 API自动使用cudnn batch normalization。
* 可以使用一个固定的参数名共享BN层的参数。
* 2D convolution operation支持variable-dimension input特性。
* 重构cmake中关于CUDA的部分并实现自动检测GPU架构的功能。
* 优化网页导航。

## 错误修复

* 修复ROI pooling的Bug. cc9a761
* 修复当label是dense vector是AUC变成0的问题. #5274
* 修复WarpCTC 层的Bug.


# v0.10.0版本

我们非常高兴发布了PaddlePaddle V0.10.0版，并开发了新的[Python API](http://research.baidu.com/paddlepaddles-new-api-simplifies-deep-learning-programs/)。

- 旧的Python API由于难以学习和使用已经过时了。使用旧版本的API至少需要两份python文件，分别是定义数据生成器和定义网络拓扑结构的文件。用户通过运行`paddle_trainer`的C++程序来启动PaddlePaddle任务，该程序调用Python解释器来运行定义网络拓扑结构的文件，然后通过迭代加载数据生成器提供的小批量数据启动训练循环。这与Python的现代编辑方式不符，比如Jupyter Notebook。

- 新版的API被称为 *V2 API*，允许我们在单个.py文件中，通过编辑更短的Python程序来定义网络结构和数据。此外，该Python程序也可以在Jupyter Notebook中运行，因为PaddlePaddle可以作为共享库来被Python程序加载和使用。

基于新的API，我们提供了一个在线的学习文档 [Deep Learning 101](http://book.paddlepaddle.org/index.en.html) 及其[中文版本](http://book.paddlepaddle.org/)。

我们还致力于迭代更新新版API的在线文档，并将新版API引入分布式集群（包括MPI和Kubernetes）训练中。我们将在下一个版本中发布更多的内容。

## 新特点

* 发布新版[Python API](http://research.baidu.com/paddlepaddles-new-api-simplifies-deep-learning-programs/)。
* 发布深度学习系列课程 [Deep Learning 101](http://book.paddlepaddle.org/index.en.html) 及其[中文版本](http://book.paddlepaddle.org/)。
* 支持矩形输入的CNN。
* 为seqlastin和seqfirstin提供stride pooling。
* 在`trainer_config_helpers`中暴露`seq_concat_layer/seq_reshape_layer`。
* 添加公共数据集包：CIFAR，MNIST，IMDB，WMT14，CONLL05，movielens，imikolov。
* 针对Single Shot Multibox Detection增加 Prior box layer。
* 增加光滑的L1损失。
* 在V2 API中增加 data reader 创建器和修饰器。
* 增加cmrnorm投影的CPU实现。


## 改进

* 提供`paddle_trainer`的Python virtualenv支持。
* 增加代码自动格式化的pre-commit hooks。
* 升级protobuf到3.x版本。
* 在Python数据生成器中提供一个检测数据类型的选项。
* 加速GPU中average层的后向反馈计算。
* 细化文档。
* 使用Travis-CI检查文档中的死链接。
* 增加解释`sparse_vector`的示例。
* 在layer_math.py中添加ReLU。
* 简化Quick Start示例中的数据处理流程。
* 支持CUDNN Deconv。
* 在v2 API中增加数据feeder。
* 在情感分析示例的演示中增加对标准输入流中样本的预测。
* 提供图像预处理的多进程接口。
* 增加V1 API的基准文档。
* 在`layer_math.py`中增加ReLU。
* 提供公共数据集的自动下载包。
* 将`Argument::sumCost`重新命名为`Argument::sum`，并暴露给python。
* 为矩阵相关的表达式评估增加一个新的`TensorExpression`实现。
* 增加延迟分配来优化批处理多表达式计算。
* 增加抽象的类函数及其实现：
  * `PadFunc` 和 `PadGradFunc`。
  * `ContextProjectionForwardFunc` 和 `ContextProjectionBackwardFunc`。
  * `CosSimBackward` 和 `CosSimBackwardFunc`。
  * `CrossMapNormalFunc` 和 `CrossMapNormalGradFunc`。
  * `MulFunc`。
* 增加`AutoCompare`和`FunctionCompare`类，使得编写比较gpu和cpu版本函数的单元测试更容易。
* 生成`libpaddle_test_main.a`并删除测试文件内的主函数。
* 支持PyDataProvider2中numpy的稠密向量。
* 清理代码库，删除一些复制粘贴的代码片段：
  * 增加`SparseRowMatrix`的抽样类`RowBuffer`。
  * 清理`GradientMachine`的接口。
  * 在layer中增加`override`关键字。
  * 简化`Evaluator::create`，使用`ClassRegister`来创建`Evaluator`。
* 下载演示的数据集时检查MD5校验。
* 添加`paddle::Error`，用于替代Paddle中的`LOG(FATAL)`。


## 错误修复

* 检查`recurrent_group`的layer输入类型。
* 不要用.cu源文件运行`clang-format`。
* 修复`LogActivation`的使用错误。
* 修复运行`test_layerHelpers`多次的错误。
* 修复seq2seq示例超出消息大小限制的错误。
* 修复在GPU模式下dataprovider转换的错误。
* 修复`GatedRecurrentLayer`中的错误。
* 修复在测试多个模型时`BatchNorm`的错误。
* 修复paramRelu在单元测试时崩溃的错误。
* 修复`CpuSparseMatrix`编译时相关的警告。
* 修复`MultiGradientMachine`在`trainer_count > batch_size`时的错误。
* 修复`PyDataProvider2`阻止异步加载数据的错误。

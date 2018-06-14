## C-API使用流程

这篇文档介绍 PaddlePaddle C-API 整体使用流程。

### 使用流程

使用 C-API 的工作流程如图1所示，分为（1）准备预测模型和（2）预测程序开发两大部分。

<p align="center">
<img src="https://user-images.githubusercontent.com/5842774/34658453-365f73ea-f46a-11e7-9b3f-0fd112b27bae.png" width=500><br> 图1. C-API使用流程示意图
</p>

- 准备预测模型

    1. 只将神经网络结构进行序列化。
        - 只对神经网络结构进行序列化，加载模型需同时指定：网络结构的序列化结果和模型参数存储目录。
    1. 将网络结构定义和训练结束存储下来的模型参数文件（多个）合并入一个文件。
        - 神经网络模型结构和训练好的模型将被序列化合并入一个文件。
        - 预测时只需加载一个文件便于发布。
    - **注意**：以上两种方式只需选择其一即可。
- 调用 C-API 开发预测序

    1. 初始化PaddlePaddle运行环境。
    1. 加载预测模型。
    1. 创建神经网络输入，组织输入数据。
    1. 进行前向计算，获得计算结果。
    1. 清理和结束。

### 准备预测模型

准备预测模型部分，我们以手写数字识别任务为例进行介绍。手写数字识别任务定义了一个含有[两个隐层的简单全连接网络](https://github.com/PaddlePaddle/book/blob/develop/02.recognize_digits/README.cn.md#softmax回归softmax-regression)，网络接受一幅图片作为输入，将图片分类到 0 ~ 9 类别标签之一。完整代码可以查看[此目录](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense) 中的相关脚本。

调用C-API开发预测程序需要一个训练好的模型，运行[MNIST手写数字识别目录](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense)下的[mnist_v2.py](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/examples/model_inference/dense/mnist_v2.py)脚本，在终端执行`python mnist_v2.py`，会使用 PaddlePaddle 内置的 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)进行训练。训练好的模型默认保存在当前运行目录下的`models`目录中。

下面，我们将训练结束后存储下来的模型转换成预测模型。

1. 序列化神经网络模型配置

    PaddlePaddle 使用 protobuf 来传输网络配置文件中定义的网络结构和相关参数，使用 C-API 进行预测时，需要将网络结构使用 protobuf 进行序列化，写入文件中。

    调用[`paddle.utils.dump_v2_config`](https://github.com/PaddlePaddle/Paddle/tree/develop/python/paddle/utils/dump_v2_config.py)中的`dump_v2_config`函数能够将使用 PaddlePaddle V2 API 定义的神经网络结构 dump 到指定文件中，示例代码如下：

    ```python
    from paddle.utils.dump_v2_config import dump_v2_config
    from mnist_v2 import network

    predict = network(is_infer=True)
    dump_v2_config(predict, "trainer_config.bin", True)
    ```

    对[手写数字识别](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense)这个示例，[`mnist_v2.py`](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense/mnist_v2.py)脚本集成了序列化神经网络结构的过程，可以直接运行 `python mnist_v2.py --task dump_config` 对神经网络结构进行序列化，结果会写入当前运行目录下的`trainer_config.bin`文件中。

    使用这种方式，需要**在运行时将神经网络的多个可学习参数放在同一个目录中**，C-API可以通过分别指定序列化后的网络结构文件和参数目录来加载训练好的模型。

2. 合并模型文件(可选)

    一些情况为了便于发布，希望能够将序列化后的神经网络结构和训练好的模型参数打包进一个文件。对于这样的需求，可以使用`paddle.utils.merge_model`中的`merge_v2_model`接口对神经网络结构和训练好的参数进行序列化，将序列化结果写入一个文件内。

    代码示例如下：

    ```python
    from paddle.utils.merge_model import merge_v2_model
    from mnist_v2 import network

    net = network(is_infer=True)
    param_file = "models/params_pass_4.tar"
    output_file = "output.paddle.model"
    merge_v2_model(net, param_file, output_file)
    ```

    对[手写数字识别](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense)这个示例，可直接运行 `python` [merge_v2_model.py](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference/dense/merge_v2_model.py)。序列化结果会写入当前运行目录下的`output.paddle.model`文件中。使用这种方式，运行时C-API可以通过指定`output.paddle.model`文件的路径来加载预测模型。

#### 注意事项
1. 为使用C-API，在调用`dump_v2_config`序列化神经网络结构时，参数`binary`必须指定为`True`。
1. **预测使用的网络结构往往不同于训练**，通常需要去掉网络中的：（1）类别标签层；（2）损失函数层；（3）`evaluator`等，只留下核心计算层，请注意是否需要修改网络结构。
1. 预测时，可以获取网络中定义的任意多个（大于等于一个）层前向计算的结果，需要哪些层的计算结果作为输出，就将这些层加入一个Python list中，作为调用`dump_v2_config`的第一个参数。

### 编写预测代码

预测代码更多详细示例代码请参考[C-API使用示例](https://github.com/PaddlePaddle/Paddle/tree/develop/paddle/capi/examples/model_inference) 目录下的代码示例。这一节对图1中预测代码编写的5个步骤进行介绍和说明。

#### step 1. 初始化PaddlePaddle运行环境
第一步需调用[`paddle_init`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/main.h#L27) 初始化PaddlePaddle运行环境，该接口接受两个参数：参数的个数和参数列表。

#### step2. 加载模型

这里介绍C-API使用中的一个重要概念：Gradient Machine。

概念上，在 PaddlePaddle 内部，一个GradientMachine类的对象管理着一组计算层（PaddlePaddle Layers）来完成前向和反向计算，并处理与之相关的所有细节。在调用C-API预测时，只需进行前向计算而无需调用反向计算。这篇文档之后部分会使用`gradient machine`来特指调用PaddlePaddle C-API创建的GradientMachine类的对象。每一个 `gradient machine` 都会管理维护一份训练好的模型，下面是C-API提供的，两种常用的模型加载方式：

1. 调用[`paddle_gradient_machine_load_parameter_from_disk`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/gradient_machine.h#L61)接口，从磁盘加载预测模型。这时`gradient machine`会独立拥有一份训练好的模型；
1. 调用[`paddle_gradient_machine_create_shared_param`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/gradient_machine.h#L88)接口，与其它`gradient machine`的共享已经加载的预测模型。这种情况多出现在使用多线程预测时，通过多个线程共享同一个模型来减少内存开销。可参考[此示例](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/examples/model_inference/multi_thread/main.c)。

- 注意事项

    1. 使用PaddlePaddle V2 API训练，模型中所有可学习参数会被存为一个压缩文件，需要手动进行解压，将它们放在同一目录中，C-API不会直接加载 V2 API 存储的压缩文件。
    1. 如果使用`merge model`方式将神经网络结构和训练好的参数序列化到一个文件，请参考此[示例](https://github.com/PaddlePaddle/Mobile/blob/develop/Demo/linux/paddle_image_recognizer.cpp#L59)。
    1. 通过灵活使用以上两个接口，加载模型可其它多种方式，例如也可在程序运行过程中再加载另外一个模型。

#### step 3. 创建神经网络输入，组织输入数据

基本使用概念：
- 在PaddlePaddle内部，神经网络中一个计算层的输入输出被组织为一个 `Argument` 结构体，如果神经网络有多个输入或者多个输出，每一个输入/输出都会对应有自己的`Argument`。
- `Argument` 并不真正“存储”数据，而是将输入/输出数据有机地组织在一起。
- 在`Argument`内部由：1. `Matrix`（二维矩阵，存储浮点类型输入/输出）；2. `IVector`（一维数组，**仅用于存储整型值**，多用于自然语言处理任务）来实际存储数据。

C-API支持的所有输入数据类型和他们的组织方式，请参考“输入/输出数据组织”一节。

这篇文档的之后部分会使用`argument`来特指PaddlePaddle C-API中神经网络的一个输入/输出，使用`paddle_matrix`**特指**`argument`中用于存储数据的`Matrix`类的对象。

在组织神经网络输入，获取输出时，需要思考完成以下工作：

1. 为每一个输入/输出创建`argument`；
1. 为每一个`argument`创建`paddle_matrix`来存储数据；

与输入不同的是，不需在使用C-API时为输出`argument`的`paddle_matrix`对象分配空间。前向计算之后PaddlePaddle内部已经分配/管理了每个计算层输出的存储空间。

#### step 4. 前向计算

完成上述准备之后，通过调用 [`paddle_gradient_machine_forward`](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/capi/gradient_machine.h#L73) 接口完成神经网络的前向计算。

#### step 5. 清理

结束预测之后，对使用的中间变量和资源进行清理和释放。

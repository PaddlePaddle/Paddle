############
数据输入实例
############

数据读取是训练和测试的一个重要环节，为了方便使用者更容易地编写数据传输的代码, PaddlePaddle提供了Reader接口，使得用户可以只关注如何从文件中读取每一条数据，而不用关心数据传输给PaddlePaddle的具体细节。

非序列数据的读取
============

MNIST
----------------------

这里以MNIST手写识别为例，来说明如何读取非序列数据。

MNIST是一个包含有 70,000张灰度图片的数字分类数据集：输入特征是 28*28 像素的灰度值，类别标签是0~9的整型值。假设使用文本文件存储原始的MNIST图片，样例数据如下：

..  literalinclude:: src/mnist_train.txt
    :linenos:

其中，每一行是一条完整的数据样本，';'分隔的两列，第一列数据为这张图片的label，第二列数据为这个图片的像素值。

那么对应的数据处理读取接口为：

..  literalinclude:: src/train_reader_mnist.py
    :linenos:

其中，train_reader函数会返回一个reader函数，reader函数是实现训练数据输入的主函数，在这个函数中，实现了打开文本文件，从文本文件中读取每一行，并将每行转换成训练配置文件中需要的格式，然后返回给PaddlePaddle进程。需要注意的是，数据各部分返回的顺序要和训练配置中定义的顺序保持一致，当不一致时需要通过feeding字典来指定trainer与data layer的对应关系。

如上述代码所示，python提供的数据读取接口是一个 python 的 generator , 用户只需要实现一条数据读取的逻辑，而不必关心数据传输给PaddlePaddle的具体细节。 其中，reader函数是使用关键词 yield 构造的 generator, 当在reader函数调用多次 yield ，可以为一个数据文件返回多条训练样本 。 yield 是Python的一个关键词，使用这个关键词可以在一个函数里多次返回变量。

对应的data layer定义如下：

..	code-block:: python

    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(784))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(10))

这里的type表示输入数据在PaddlePaddle中的类型，详细信息可以查看 `基本实用概念 <../concepts/use_concepts_cn.html>`_ 中的数据类型进行了解。

在训练配置里，reader是train接口的一个参数：

..  literalinclude:: src/train_mnist.py
    :linenos:

这里feeding是一个字典，定义了数据传入的顺序信息，其中key对应传入data layer的名称，value表示train_reader函数返回的迭代器中的一条样本各部分对应到data layer的位置映射信息。 当reader返回的数据各部分的顺序和训练配置中data layer定义的顺序一致时，feeding可以不用设置。否则，当返回数据的各部分和data layer之间不是一一对应的关系，或者两者顺序不一致，我们需要定义feeding字典来指定两者具体的映射关系。

序列模型数据的读取
----------------------

序列模型数据是指数据的某一维度是一个序列形式，即包含时间步信息。所谓时间步信息，不一定和时间有关系，只是说明数据的顺序是重要的。例如，文本信息就是一个序列数据。这里以英文情感分类数据为例，来说明如何读取非序列数据。

在情感分类任务中，给定一段输入的文本和文本反映出的情感标签，在这个例子中我们只考虑正面情绪和负面情绪两类，分别用0和1编号。下面是两条示例数据：

..  literalinclude:: src/sentimental_train.txt
    :linenos:

其中，每一行是一条完整的数据样本，';'分隔的两列，第一列数据是该行文本对应的label，第二列数据是文本的具体内容。

对应的数据读取函数为：

..  literalinclude:: src/train_reader_seq.py
    :linenos:

和上述的MNIST数据处理有所不同，序列数据输入类型不再是 :code:`dense_vector` ，而变成了 :code:`integer_value_sequence`。 对于文本数据的处理，需要一个词典来将输入的每个token映射为词典的索引。因此，这里的输入特征是词id的序列。

对应的data layer的配置如下：

..	code-block:: python

    x = paddle.layer.data(name='x',
                      type=paddle.data_type.integer_value_sequence(len(word_dict)))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(2))

在训练配置里，将reader传入train接口：

..  literalinclude:: src/train_seq.py
    :linenos:

这里和MNIST数据大体类似，除了对于词典需要进行预处理，并将其作为参数传入train_reader中。此外，需要注意的是，这里我们输入数据是序列类型，因此对应data layer的input_type需要设置成integer_value_sequence格式。而在reader函数中，基本的处理逻辑也和mnist数据处理方式一致，依次返回了文件中的每条数据。

############
数据输入实例
############

在训练和测试的过程中，PaddlePaddle需要读取数据。为了方便使用者更容易的编写数据传输的代码, PaddlePaddle提供了Reader接口，使用该接口用户可以只关注如何从文件中读取每一条数据，而不用关心数据传输给PaddlePaddle的具体细节。


简单的使用场景
============

MNIST
----------------------

这里以MNIST手写识别为例，来说明简单的Reader接口如何使用。MNIST是一个包含有 70,000张灰度图片的数字分类数据集。对于MNIST而言，标签是0-9的数字，而特征即为 28*28的像素灰度值。这里我们使用简单的文本文件表示MNIST图片，样例数据如下。

..  literalinclude:: src/mnist_train.txt
    :linenos:

其数据使用;间隔，第一段数据为这张图片的label，第二段数据为这个图片的像素值。

那么对应的数据处理读取为：

..  literalinclude:: src/train_reader_mnist.py
    :linenos:

其中，train_reader函数会返回一个reader函数，reader函数是实现训练数据输入的主函数，在这个函数中，实现了打开文本文件，从文本文件中读取每一行，并将每行转换成训练配置文件中需要的格式，然后返回给PaddlePaddle进程。需要注意的是，返回的顺序需要和训练配置中定义的顺序一致。

同时，返回数据在PaddlePaddle中是仅仅返回一条完整的训练样本。 通过使用关键词 yield ， 可以为一个数据文件返回多条训练样本(就像这个样例一样)，只需要在reader函数调用多次 yield 即可。 yield 是Python的一个关键词，相关的概念是 generator 。使用这个关键词，可以在一个函数里，多次返回变量。

对应的数据层可以定义如下：

..	code-block:: python

    x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(784))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(10))
    
这里的type表示输入数据的具体格式，详细信息可以查看 `基本实用概念 <../concepts/use_concepts_cn.html>`_ 中的数据类型进行了解。
    
在训练配置里，可以很方便的设置训练引用这个reader。这个设置为：

..  literalinclude:: src/train_mnist.py
    :linenos:

这里feeding是一个字典，定义了数据传入的顺序信息，其中key对应传入数据层的名称，value表示train_reader函数返回的迭代器中的一条样本各部分对应到数据层的位置映射信息。 


序列模型数据
----------------------

序列模型数据是指数据的某一维度是一个序列形式，即包含时间步信息。所谓时间步信息，不一定和时间有关系，只是说明数据的顺序是重要的。例如，文本信息就是一个序列数据。

这里举例的数据是英文情感分类的数据。数据是给一段英文文本，分类成正面情绪和负面情绪两类(用0和1表示)。样例数据为：

..  literalinclude:: src/sentimental_train.txt
    :linenos:

对应的数据读取函数为：

..  literalinclude:: src/train_reader_seq.py
    :linenos:

和上述的MNIST数据处理有所不同，序列数据需要一个词典来将每个输入的每个token映射为词典的索引。因此，这里的输入特征是词id的序列。对应的数据层的配置如下：

..	code-block:: python

    x = paddle.layer.data(name='x', 
                      type=paddle.data_type.integer_value_sequence(len(word_dict)))
    y = paddle.layer.data(name='y', type=paddle.data_type.integer_value(2))


最终传入这个变量的方式为：

..  literalinclude:: src/train_seq.py
    :linenos:
    
这里和MNIST数据大体类似，除了对于词典需要进行预处理，并将其作为参数传入train_reader中。此外，需要注意的是，这里我们输入的类型是序列，因此对应数据层的input_type需要设置成integer_value_sequence格式。而在reader函数中，基本的处理逻辑也和mnist逻辑一致，依次返回了文件中的每条数据。

    
    
    



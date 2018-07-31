.. _user_guide_test_while_training:

##################
训练过程中评测模型
##################

模型的测试评价与训练的 :code:`fluid.Program` 不同。在测试评价中:

1. 评价测试不进行反向传播，不优化更新参数。
2. 评价测试执行的操作可以不同。

   * 例如 BatchNorm 操作，在训练和测试时执行不同的算法。

   * 评价模型与训练相比可以是完全不同的模型。

生成测试 :code:`fluid.Program`
#################################

通过克隆训练 :code:`fluid.Program` 生成测试 :code:`fluid.Program`
=======================================================================

:code:`Program.clone()` 方法可以复制出新的 :code:`fluid.Program` 。 通过设置
:code:`Program.clone(for_test=True)` 复制含有用于测试的操作Program。简单的使用方法如下:

.. code-block:: python

   import paddle.fluid as fluid

   img = fluid.layers.data(name="image", shape=[784])
   prediction = fluid.layers.fc(
     input=fluid.layers.fc(input=img, size=100, act='relu'),
     size=10,
     act='softmax'
   )
   label = fluid.layers.data(name="label", shape=[1], dtype="int64")
   loss = fluid.layers.mean(fluid.layers.cross_entropy(input=prediction, label=label))
   acc = fluid.layers.accuracy(input=prediction, label=label)

   test_program = fluid.default_main_program().clone(for_test=True)

   adam = fluid.optimizer.Adam(learning_rate=0.001)
   adam.minimize(loss)

在使用 :code:`Optimizer` 之前，将 :code:`fluid.default_main_program()` 复制\
成一个 :code:`test_program` 。之后使用测试数据运行 :code:`test_program`,\
就可以做到运行测试程序，而不影响训练结果。

分别配置训练 :code:`fluid.Program` 和测试 :code:`fluid.Program`
=====================================================================

如果训练程序和测试程序相差较大时，用户也可以通过完全定义两个不同的
:code:`fluid.Program`，分别进行训练和测试。在PaddlePaddle Fluid中，\
所有的参数都有名字。如果两个不同的操作，甚至两个不同的网络使用了同样名字的参数，\
那么他们的值和内存空间都是共享的。

PaddlePaddle Fluid中使用 :code:`fluid.unique_name` 包来随机初始化用户未定义的\
参数名称。通过 :code:`fluid.unique_name.guard` 可以确保多次调用某函数\
参数初始化的名称一致。

例如:

.. code-block:: python

   import paddle.fluid as fluid

   def network(is_test):
       file_obj = fluid.layers.open_files(filenames=["test.recordio"] if is_test else ["train.recordio"], ...)
       img, label = fluid.layers.read_file(file_obj)
       hidden = fluid.layers.fc(input=img, size=100, act="relu")
       hidden = fluid.layers.batch_norm(input=hidden, is_test=is_test)
       ...
       return loss

   with fluid.unique_name.guard():
       train_loss = network(is_test=False)
       sgd = fluid.optimizer.SGD(0.001)
       sgd.minimize(train_loss)

   test_program = fluid.Program()
   with fluid.unique_name.guard():
       with fluid.program_gurad(test_program, fluid.Program()):
           test_loss = network(is_test=True)

   # fluid.default_main_program() is the train program
   # fluid.test_program is the test program

执行测试 :code:`fluid.Program`
#################################

使用 :code:`Executor` 执行测试 :code:`fluid.Program`
=======================================================

用户可以使用 :code:`Executor.run(program=...)` 来执行测试
:code:`fluid.Program`。

例如

.. code-block:: python

   exe = fluid.Executor(fluid.CPUPlace())
   test_acc = exe.run(program=test_program, feed=test_data_batch, fetch_list=[acc])
   print 'Test accuracy is ', test_acc

使用 :code:`ParallelExecutor` 执行测试 :code:`fluid.Program`
===============================================================

用户可以使用训练用的 :code:`ParallelExecutor` 与测试 :code:`fluid.Program`
一起新建一个测试的 :code:`ParallelExecutor` ；再使用测试
:code:`ParallelExecutor.run` 来执行测试。

例如:

.. code-block:: python

   train_exec = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)

   test_exec = fluid.ParallelExecutor(use_cuda=True, share_vars_from=train_exec,
                                      main_program=test_program)
   test_acc = test_exec.run(fetch_list=[acc], ...)


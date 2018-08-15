########
单机训练
########

准备工作
########

要进行PaddlePaddle Fluid单机训练，需要先 :ref:`user_guide_prepare_data` 和
:ref:`user_guide_configure_simple_model` 。当\
:ref:`user_guide_configure_simple_model` 完毕后，可以得到两个\
:code:`fluid.Program`， :code:`startup_program` 和 :code:`main_program`。
默认情况下，可以使用 :code:`fluid.default_startup_program()` 与\ :code:`fluid.default_main_program()` 获得全局的 :code:`fluid.Program`。

例如:

.. code-block:: python

   import paddle.fluid as fluid

   image = fluid.layers.data(name="image", shape=[784])
   label = fluid.layers.data(name="label", shape=[1])
   hidden = fluid.layers.fc(input=image, size=100, act='relu')
   prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
   loss = fluid.layers.mean(
       fluid.layers.cross_entropy(
           input=prediction,
           label=label
       )
   )

   sgd = fluid.optimizer.SGD(learning_rate=0.001)
   sgd.minimize(loss)

   # Here the fluid.default_startup_program() and fluid.default_main_program()
   # has been constructed.

在上述模型配置执行完毕后， :code:`fluid.default_startup_program()` 与\
:code:`fluid.default_main_program()` 配置完毕了。

初始化参数
##########

参数随机初始化
==============

用户配置完模型后，参数初始化操作会被写入到\
:code:`fluid.default_startup_program()` 中。使用 :code:`fluid.Executor()` 运行
这一程序，即可在全局 :code:`fluid.global_scope()` 中随机初始化参数。例如:

.. code-block:: python

   exe = fluid.Executor(fluid.CUDAPlace(0))
   exe.run(program=fluid.default_startup_program())

值得注意的是: 如果使用多GPU训练，参数需要先在GPU0上初始化，再经由\
:code:`fluid.ParallelExecutor` 分发到多张显卡上。


载入预定义参数
==============

在神经网络训练过程中，经常会需要载入预定义模型，进而继续进行训练。\
如何载入预定义参数，请参考 :ref:`user_guide_save_load_vars`。


单卡训练
########

执行单卡训练可以使用 :code:`fluid.Executor()` 中的 :code:`run()` 方法，运行训练\
:code:`fluid.Program` 即可。在运行的时候，用户可以通过 :code:`run(feed=...)`\
参数传入数据；用户可以通过 :code:`run(fetch=...)` 获取持久的数据。例如:\

.. code-block:: python

   ...
   loss = fluid.layers.mean(...)

   exe = fluid.Executor(...)
   # the result is an numpy array
   result = exe.run(feed={"image": ..., "label": ...}, fetch_list=[loss])

这里有几点注意事项:

1. feed的数据格式，请参考文章 :ref:`user_guide_feed_data_to_executor`。
2. :code:`Executor.run` 的返回值是 :code:`fetch_list=[...]` 的variable值。被fetch\
   的Variable必须是persistable的。 :code:`fetch_list` 可以传入Variable的列表，\
   也可以传入Variable的名字列表。:code:`Executor.run` 返回Fetch结果列表。
3. 如果需要取回的数据包含序列信息，可以设置
   :code:`exe.run(return_numpy=False, ...)` 直接返回 :code:`fluid.LoDTensor`
   。用户可以直接访问 :code:`fluid.LoDTensor` 中的信息。

多卡训练
########

执行多卡训练可以使用 :code:`fluid.ParallelExecutor` 运行训练
:code:`fluid.Program`。例如:

.. code-block:: python

   train_exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name,
                                main_program=fluid.default_main_program())
   train_exe.run(fetch_list=[loss.name], feed={...})

这里有几点注意事项:

1. :code:`ParallelExecutor` 的构造函数需要指明要执行的 :code:`fluid.Program` ,
   并在执行过程中不能修改。默认值是 :code:`fluid.default_main_program()` 。
2. :code:`ParallelExecutor` 需要明确指定是否使用 CUDA 显卡进行训练。在显卡训练\
   模式下会占用全部显卡。用户可以配置 `CUDA_VISIBLE_DEVICES <http://www.acceleware.com/blog/cudavisibledevices-masking-gpus>`_ 来修改占用\
   的显卡。

进阶使用
########

.. toctree::
   :maxdepth: 2

   test_while_training
   save_load_variables

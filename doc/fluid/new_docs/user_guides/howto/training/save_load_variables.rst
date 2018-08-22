.. _user_guide_save_load_vars:

##################
保存与载入模型变量
##################

模型变量分类
############

在PaddlePaddle Fluid中，所有的模型变量都用 :code:`fluid.Variable()` 作为基类进行表示。
在该基类之下，模型变量主要可以分为以下几种类别：

1. 模型参数
  模型参数是深度学习模型中被训练和学习的变量，在训练过程中，训练框架根据反向传播算法计算出每一个模型参数当前的梯度，
  并用优化器根据梯度对参数进行更新。模型的训练过程本质上可以看做是模型参数不断迭代更新的过程。
  在PaddlePaddle Fluid中，模型参数用 :code:`fluid.framework.Parameter` 来表示，
  这是一个 :code:`fluid.Variable()` 的派生类，除了 :code:`fluid.Variable()` 具有的各项性质以外，
  :code:`fluid.framework.Parameter` 还可以配置自身的初始化方法、更新率等属性。

2. 长期变量
  长期变量指的是在整个训练过程中持续存在、不会因为一个迭代的结束而被销毁的变量，例如动态调节的全局学习率等。
  在PaddlePaddle Fluid中，长期变量通过将 :code:`fluid.Variable()` 的 :code:`persistable`
  属性设置为 :code:`True` 来表示。所有的模型参数都是长期变量，但并非所有的长期变量都是模型参数。

3. 临时变量
  不属于上面两个类别的所有模型变量都是临时变量，这种类型的变量只在一个训练迭代中存在，在每一个迭代结束后，
  所有的临时变量都会被销毁，然后在下一个迭代开始之前，又会先构造出新的临时变量供本轮迭代使用。
  一般情况下模型中的大部分变量都属于这一类别，例如输入的训练数据、一个普通的layer的输出等等。



如何保存模型变量
################

根据用途的不同，我们需要保存的模型变量也是不同的。例如，如果我们只是想保存模型用来进行以后的预测，
那么只保存模型参数就够用了。但如果我们需要保存一个checkpoint以备将来恢复训练，
那么我们应该将各种长期变量都保存下来，甚至还需要记录一下当前的epoch和step的id。
因为一些模型变量虽然不是参数，但对于模型的训练依然必不可少。

因此，根据需求的不同，我们提供了两套API来分别进行模型的参数和checkpoint的保存。

保存模型用于对新样本的预测
==========================

如果我们保存模型的目的是用于对新样本的预测，那么只保存模型参数就足够了。我们可以使用
:code:`fluid.io.save_params()` 接口来进行模型参数的保存。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.save_params(executor=exe, dirname=param_path, main_program=None)

上面的例子中，通过调用 :code:`fluid.io.save_params` 函数，PaddlePaddle Fluid会对默认
:code:`fluid.Program` 也就是 :code:`prog` 中的所有模型变量进行扫描，
筛选出其中所有的模型参数，并将这些模型参数保存到指定的 :code:`param_path` 之中。


保存checkpoint用于将来恢复训练
==============================

在训练过程中，我们可能希望在一些节点上将当前的训练状态保存下来，
以便在将来需要的时候恢复训练环境继续进行训练。这一般被称作“checkpoint”。
想要保存checkpoint，可以使用 :code:`fluid.io.save_checkpiont()` 接口。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./checkpoints"
    prog = fluid.default_main_program()
    trainer_args = {"epoch_id": 200,
                    "step_id": 20} # just an example
    fluid.io.save_checkpoint(executor=exe,
                                checkpoint_dir=path,
                                trainer_id=0,
                                trainer_args=trainer_args,
                                main_program=prog,
                                max_num_checkpoints=3)

上面的例子中，通过调用 :code:`fluid.io.save_checkpoint` 函数，PaddlePaddle Fluid会对默认
:code:`fluid.Program` 也就是 :code:`prog` 中的所有模型变量进行扫描，
根据一系列内置的规则自动筛选出其中所有需要保存的变量，并将他们保存到指定的 :code:`path` 目录下。

:code:`fluid.io.save_checkpoint` 的各个参数中， :code:`trainer_id` 在单机情况下设置为0即可； :code:`trainer_args`
为一个Python dict，用于给定当前的epoch_id和step_id；
:code:`max_num_checkpoints` 用于表示的最大checkpoint数量，
如果目录中已经存在的checkpoint数量超过这个值，那最早的checkpoint将被删除。

如何载入模型变量
################

与模型变量的保存相对应，我们提供了两套API来分别载入模型的参数和载入模型的checkpoint。

载入模型用于对新样本的预测
==========================

对于通过 :code:`fluid.io.save_params` 保存的模型，可以使用 :code:`fluid.io.load_params`
来进行载入。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    param_path = "./my_paddle_model"
    prog = fluid.default_main_program()
    fluid.io.load_params(executor=exe, dirname=param_path,
                         main_program=prog)

上面的例子中，通过调用 :code:`fluid.io.load_params` 函数，PaddlePaddle Fluid会对
:code:`prog` 中的所有模型变量进行扫描，筛选出其中所有的模型参数，
并尝试从 :code:`param_path` 之中读取加载它们。

需要格外注意的是，这里的 :code:`prog` 必须和调用 :code:`fluid.io.save_params`
时所用的 :code:`prog` 中的前向部分完全一致，且不能包含任何参数更新的操作。如果两者存在不一致，
那么可能会导致一些变量未被正确加载；如果错误地包含了参数更新操作，那可能会导致正常预测过程中参数被更改。
这两个 :code:`fluid.Program` 之间的关系类似于训练 :code:`fluid.Program`
和测试 :code:`fluid.Program` 之间的关系，详见： :ref:`user_guide_test_while_training`。

另外，需特别注意运行 :code:`fluid.default_startup_program()` 必须在调用 :code:`fluid.io.load_params`
之前。如果在之后运行，可能会覆盖已加载的模型参数导致错误。


载入checkpoint用于恢复训练
==========================

对于通过 :code:`fluid.io.save_checkpoint` 保存的模型，可以使用 :code:`fluid.io.load_checkpoint`
来进行载入。

例如：

.. code-block:: python

    import paddle.fluid as fluid

    exe = fluid.Executor(fluid.CPUPlace())
    path = "./checkpoints"
    prog = fluid.default_main_program()
    fluid.io.load_checkpoint(executor=exe, checkpoint_dir=path,
                             serial=9, main_program=prog)

上面的例子中，通过调用 :code:`fluid.io.save_checkpoint` 函数，PaddlePaddle Fluid会对
:code:`prog` 中的所有模型变量进行扫描，根据内置规则自动筛选出需要加载的变量，
并尝试从 :code:`path` 之中加载它们。

参数 :code:`serial` 用来标记具体要加载的checkpoint的版本号。在保存checkpoint的时候，
一个checkpoint会被保存在一个子目录中，并在目录名上体现出自己的版本号。
一般越大的版本号表示这个checkpoint越新。

这里的 :code:`prog` 必须和调用 :code:`fluid.io.save_checkpoint` 时所用的 :code:`prog`
完全一致，否则会导致变量加载错误或者未加载。另外，与 :code:`fluid.io.save_params` 类似，
运行 :code:`fluid.default_startup_program()` 也必须在 :code:`fluid.io.load_checkpoint`
之前进行。

多机checkpoint保存
##################

.. toctree::
   :maxdepth: 2

   checkpoint_doc_cn.md
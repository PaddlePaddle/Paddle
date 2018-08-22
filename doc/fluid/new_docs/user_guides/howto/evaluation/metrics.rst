############
模型评估
############

模型评估是用指标反映模型在预期目标下精度，根据模型任务决定观察指标，作为在训练中调整超参数，评估模型效果的重要依据。
metric函数的输入为当前模型的预测preds和labels，输出是自定义的。metric函数和loss函数非常相似，但是metric并不是模型训练网络组成部分。

用户可以通过训练网络得到当前的预测preds和labels，在Python端定制metric函数；也可以通过定制c++ Operator的方式，在GPU上加速metric计算。

paddle.fluid.metrics模块包含该功能


常用指标
############

metric函数根据模型任务不同，指标构建方法因任务而异。

回归类型任务labels是实数，因此loss和metric函数构建相同，可参考MSE的方法。
分类任务常用指标为分类指标，本文提到的一般是二分类指标，多分类和多标签需要查看对应的API文档。例如排序指标auc，多分类可以作为0，1分类任务，auc指标仍然适用。
Fluid中包含了常用分类指标，例如Precision, Recall, Accuracy等,更多请阅读API文档。以 :ref:`Precision` 为例，具体方法为

.. code-block:: python

   >>> import paddle.fluid as fluid
   >>> labels = fluid.layers.data(name="data", shape=[1], dtype="int32")
   >>> data = fluid.layers.data(name="data", shape=[32, 32], dtype="int32")
   >>> pred = fluid.layers.fc(input=data, size=1000, act="tanh")
   >>> acc = fluid.metrics.Precision()
   >>> for pass in range(PASSES):
   >>>   acc.reset()
   >>>   for data in train_reader():
   >>>       loss, preds, labels = exe.run(fetch_list=[cost, preds, labels])
   >>>   acc.update(preds=preds, labels=labels)
   >>>   numpy_acc = acc.eval()
      

其他任务例如MultiTask Learning，Metric Learning，Learning To Rank各种指标构造方法请参考API文档。

自定义指标
############
Fluid支持自定义指标，灵活支持各类计算任务。下文通过一个简单的计数器metric函数，实现对模型的评估。
其中preds是模型预测值，labels是给定的标签。

.. code-block:: python

   >>> class MyMetric(MetricBase):
   >>>     def __init__(self, name=None):
   >>>         super(MyMetric, self).__init__(name)
   >>>         self.counter = 0  # simple counter

   >>>     def reset(self):
   >>>         self.counter = 0

   >>>     def update(self, preds, labels):
   >>>         if not _is_numpy_(preds):
   >>>             raise ValueError("The 'preds' must be a numpy ndarray.")
   >>>         if not _is_numpy_(labels):
   >>>             raise ValueError("The 'labels' must be a numpy ndarray.")
   >>>         self.counter += sum(preds == labels)

   >>>     def eval(self):
   >>>         return self.counter

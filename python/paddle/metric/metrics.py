#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['Metric', 'Accuracy']

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import abc
#import numpy as np
import paddle.fluid as fluid


@six.add_metaclass(abc.ABCMeta)
class Metric(object):
    """
    Base class for metric, encapsulates metric logic and APIs
    Usage:
        
        m = SomeMetric()
        for prediction, label in ...:
            m.update(prediction, label)
        m.accumulate()
        
    Advanced usage for :code:`compute`:

    Metric calculation can be accelerated by calculating metric states
    from model outputs and labels by build-in operators not by Python/Numpy
    in :code:`compute`, metric states will be fetched as numpy array and
    call :code:`update` with states in numpy format.
    Metric calculated as follows (operations in Model and Metric are
    indicated with curly brackets, while data nodes not):
                 inputs & labels              || ------------------
                       |                      ||
                    {model}                   ||
                       |                      ||
                outputs & labels              ||
                       |                      ||    tensor data
             {Metric.compute}           ||
                       |                      ||
              metric states(tensor)           ||
                       |                      ||
                {fetch as numpy}              || ------------------
                       |                      ||
              metric states(numpy)            ||    numpy data
                       |                      ||
                {Metric.update}               \/ ------------------
    Examples:
        
        For :code:`Accuracy` metric, which takes :code:`pred` and :code:`label`
        as inputs, we can calculate the correct prediction matrix between
        :code:`pred` and :code:`label` in :code:`compute`.
        For examples, prediction results contains 10 classes, while :code:`pred`
        shape is [N, 10], :code:`label` shape is [N, 1], N is mini-batch size,
        and we only need to calculate accurary of top-1 and top-5, we could
        calculate the correct prediction matrix of the top-5 scores of the
        prediction of each sample like follows, while the correct prediction
        matrix shape is [N, 5].

        .. code-block:: python
            def compute(pred, label):
                # sort prediction and slice the top-5 scores
                pred = fluid.layers.argsort(pred, descending=True)[1][:, :5]
                # calculate whether the predictions are correct
                correct = pred == label
                return fluid.layers.cast(correct, dtype='float32')

        With the :code:`compute`, we split some calculations to OPs (which
        may run on GPU devices, will be faster), and only fetch 1 tensor with
        shape as [N, 5] instead of 2 tensors with shapes as [N, 10] and [N, 1].
        :code:`update` can be define as follows:

        .. code-block:: python
            def update(self, correct):
                accs = []
                for i, k in enumerate(self.topk):
                    num_corrects = correct[:, :k].sum()
                    num_samples = len(correct)
                    accs.append(float(num_corrects) / num_samples)
                    self.total[i] += num_corrects
                    self.count[i] += num_samples
                return accs
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def reset(self):
        """
        Reset states and result
        """
        raise NotImplementedError("function 'reset' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def update(self, *args):
        """
        Update states for metric

        Inputs of :code:`update` is the outputs of :code:`Metric.compute`,
        if :code:`compute` is not defined, the inputs of :code:`update`
        will be flatten arguments of **output** of mode and **label** from data:
        :code:`update(output1, output2, ..., label1, label2,...)`

        see :code:`Metric.compute`
        """
        raise NotImplementedError("function 'update' not implemented in {}.".
                                  format(self.__class__.__name__))

    @abc.abstractmethod
    def accumulate(self):
        """
        Accumulates statistics, computes and returns the metric value
        """
        raise NotImplementedError(
            "function 'accumulate' not implemented in {}.".format(
                self.__class__.__name__))

    @abc.abstractmethod
    def name(self):
        """
        Returns metric name
        """
        raise NotImplementedError("function 'name' not implemented in {}.".
                                  format(self.__class__.__name__))

    def compute(self, *args):
        """
        This API is advanced usage to accelerate metric calculating, calulations
        from outputs of model to the states which should be updated by Metric can
        be defined here, where Paddle OPs is also supported. Outputs of this API
        will be the inputs of "Metric.update".

        If :code:`compute` is defined, it will be called with **outputs**
        of model and **labels** from data as arguments, all outputs and labels
        will be concatenated and flatten and each filed as a separate argument
        as follows:
        :code:`compute(output1, output2, ..., label1, label2,...)`

        If :code:`compute` is not defined, default behaviour is to pass
        input to output, so output format will be:
        :code:`return output1, output2, ..., label1, label2,...`

        see :code:`Metric.update`
        """
        return args


class Accuracy(Metric):
    """
    Encapsulates accuracy metric logic

    Example by standalone:
        
        .. code-block:: python

        import numpy as np
        import paddle

        paddle.enable_imperative()
        
        x = paddle.imperative.to_variable(np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.3, 0.2],
            [0.1, 0.2, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.4]]))
        y = paddle.imperative.to_variable(np.array([[0], [1], [2], [3]]))

        m = paddle.incubate.hapi.metrics.Accuracy()
        correct = m.compute(x, y)
        m.update(correct)
        res = m.accumulate()
        print(res)


    Example with Model API:
        
        .. code-block:: python

        import paddle
        import paddle.fluid as fluid
        import paddle.incubate.hapi as hapi

        fluid.enable_dygraph()

        train_dataset = hapi.datasets.MNIST(mode='train')

        model = hapi.Model(hapi.vision.LeNet(classifier_activation=None))
        optim = fluid.optimizer.Adam(
            learning_rate=0.001, parameter_list=model.parameters())
        model.prepare(
            optim,
            loss_function=paddle.nn.CrossEntropyLoss(),
            metrics=paddle.metrics.Accuracy())

        model.fit(train_dataset, batch_size=64)

    """

    def __init__(self, topk=(1, ), name=None, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)
        self.topk = topk
        self.maxk = max(topk)
        self._init_name(name)
        self.reset()

    def compute(self, pred, label, *args):
        pred = fluid.layers.argsort(pred, descending=True)[1][:, :self.maxk]
        correct = pred == label
        return fluid.layers.cast(correct, dtype='float32')

    def update(self, correct, *args):
        if isinstance(correct, fluid.core.VarBase):
            correct = correct.numpy()
        accs = []
        for i, k in enumerate(self.topk):
            num_corrects = correct[:, :k].sum()
            num_samples = len(correct)
            accs.append(float(num_corrects) / num_samples)
            self.total[i] += num_corrects
            self.count[i] += num_samples
        return accs

    def reset(self):
        self.total = [0.] * len(self.topk)
        self.count = [0] * len(self.topk)

    def accumulate(self):
        res = []
        for t, c in zip(self.total, self.count):
            res.append(float(t) / c)
        return res

    def _init_name(self, name):
        name = name or 'acc'
        if self.maxk != 1:
            self._name = ['{}_top{}'.format(name, k) for k in self.topk]
        else:
            self._name = [name]

    def name(self):
        return self._name

#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
"""
Testing and training events.

There are:

* TestResult
* BeginIteration
* EndIteration
* BeginPass
* EndPass
"""
__all__ = [
    'EndIteration', 'BeginIteration', 'BeginPass', 'EndPass', 'TestResult',
    'EndForwardBackward'
]


class WithMetric(object):
    def __init__(self, evaluator):
        import py_paddle.swig_paddle as api
        if not isinstance(evaluator, api.Evaluator):
            raise TypeError("Evaluator should be api.Evaluator type")
        self.__evaluator__ = evaluator

    @property
    def metrics(self):
        names = self.__evaluator__.getNames()
        retv = dict()
        for each_name in names:
            val = self.__evaluator__.getValue(each_name)
            retv[each_name] = val
        return retv


class TestResult(WithMetric):
    """
    Result that trainer.test return.
    """

    def __init__(self, evaluator, cost):
        super(TestResult, self).__init__(evaluator)
        self.cost = cost


class BeginPass(object):
    """
    Event On One Pass Training Start.
    """

    def __init__(self, pass_id):
        self.pass_id = pass_id


class EndPass(WithMetric):
    """
    Event On One Pass Training Complete.
    To get the output of a specific layer, add "event.gm.getLayerOutputs('predict_layer')"
    in your event_handler call back
    """

    def __init__(self, pass_id, evaluator, gm):
        self.pass_id = pass_id
        self.gm = gm
        WithMetric.__init__(self, evaluator)


class BeginIteration(object):
    """
    Event On One Batch Training Start.
    """

    def __init__(self, pass_id, batch_id):
        self.pass_id = pass_id
        self.batch_id = batch_id


class EndForwardBackward(object):
    """
    Event On One Batch ForwardBackward Complete.
    """

    def __init__(self, pass_id, batch_id, gm):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.gm = gm


class EndIteration(WithMetric):
    """
    Event On One Batch Training Complete.
    To get the output of a specific layer, add "event.gm.getLayerOutputs('predict_layer')"
    in your event_handler call back
    """

    def __init__(self, pass_id, batch_id, cost, evaluator, gm):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost
        self.gm = gm
        WithMetric.__init__(self, evaluator)

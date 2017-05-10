"""
Testing and training events.

There are:

* TestResult
* BeginIteration
* EndIteration
* BeginPass
* EndPass
"""
import py_paddle.swig_paddle as api

__all__ = [
    'EndIteration', 'BeginIteration', 'BeginPass', 'EndPass', 'TestResult'
]


class WithMetric(object):
    def __init__(self, evaluator):
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
    """

    def __init__(self, pass_id, evaluator):
        self.pass_id = pass_id
        WithMetric.__init__(self, evaluator)


class BeginIteration(object):
    """
    Event On One Batch Training Start.
    """

    def __init__(self, pass_id, batch_id):
        self.pass_id = pass_id
        self.batch_id = batch_id


class EndIteration(WithMetric):
    """
    Event On One Batch Training Complete.
    """

    def __init__(self, pass_id, batch_id, cost, evaluator):
        self.pass_id = pass_id
        self.batch_id = batch_id
        self.cost = cost
        WithMetric.__init__(self, evaluator)

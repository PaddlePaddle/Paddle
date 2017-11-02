import paddle.v2.framework.op as op
import numpy as np
import paddle.v2.framework.core as core


class Evaluator(object):
    """
    Evalutor Base class.
    """

    def __init__(self):
        """
       create metric states and append to block
       """
        pass

    def _clear_state(self):
        """
      clear metric states at the begin of each pass
      """
        pass

    def _append_evalutor_op(self):
        """
      add mini-batch caculate operators to block
      add increment operator to accumulate the metric state
      """
        pass

    def _merge(self):
        """
      Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
      """
        pass

    def evaluate(self):
        """
      only one exported interface
      user calculate the result
      """
        pass

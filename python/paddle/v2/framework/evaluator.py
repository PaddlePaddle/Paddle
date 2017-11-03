from paddle.v2.framework.framework import Program, unique_name
from paddle.v2.framework.layer_helper import LayerHelper
import paddle.v2.framework.core as core


class Evaluator(object):
    """
    Evalutor Base class.

    create metric states
    add mini-batch evaluator caculate operator
    add increment operator to accumulate the metric states
    """

    def __init__(self, evaluator_type, **kwargs):
        self._states = []
        self._helper = LayerHelper(layer_type=evaluator_type, **kwargs)

    @staticmethod
    def clear(self):
        """
      clear metric states at the begin of each pass/user specified batch
      """
        raise NotImplementedError()

    def evaluate(self):
        """
      Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
      """
        raise NotImplementedError()


class Accuracy(Evaluator):
    def __init__(self, input, label, k=1, **kwargs):
        super(Accuracy, self).__init__("accuracy", **kwargs)
        g_total = helper.create_global_variable(
            name=unique_name("Total"),
            persistable=True,
            dtype="int64",
            shape=[1])
        g_correct = helper.create_global_variable(
            name=unique_name("Correct"),
            persistable=True,
            dtype="int64",
            shape=[1])

        topk_out = helper.create_tmp_variable(dtype=input.data_type)
        topk_indices = helper.create_tmp_variable(dtype="int64")
        helper.append_op(
            type="top_k",
            inputs={"X": [input]},
            outputs={"Out": [topk_out],
                     "Indices": [topk_indices]},
            attrs={"k": k})
        acc_out_dtype = kwargs.get("out_dtype", "float32")
        acc_out = helper.create_tmp_variable(dtype=acc_out_dtype)
        helper.append_op(
            type="accuracy",
            inputs={
                "Out": [topk_out],
                "Indices": [topk_indices],
                "Label": [label]
            },
            outputs={"Accuracy": [acc_out]})

        helper.append_op(
            type="sum", inputs={"X": [g_total, ], },
            outputs={"Out": [g_total]})

        return acc_out

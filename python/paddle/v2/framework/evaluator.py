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

    def __init__(self, name, **kwargs):
        self._states = []
        self._helper = LayerHelper(layer_type=name, **kwargs)

    # def _update(self):
    #     """
    #     Updates the internal states througth operator
    #   """
    #     raise NotImplementedError()

    def reset(self):
        """
      Clear metric states at the begin of each pass/user specified batch
      """
        reset_program = Program()
        for var in self._states:
            zeros = helper.create_tmp_variable(dtype=var.data_type)
            self._helper.append_op(
                type="fill_constant",
                outputs={"Out": [zeros]},
                attrs={
                    "shape": var.shape,
                    "value": 0,
                })
            self._helper.append_op(
                type="scale", inputs={"X": zeros}, outputs={"Out": var})
        return reset_program

    def eval(self):
        """
      Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
      """
        raise NotImplementedError()


class Accuracy(Evaluator):
    """
    Accuracy need two state variable Total, Correct
    """

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
        self._states.append(g_total)
        self._states.append(g_correct)

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
            outputs={
                "Accuracy": [acc_out],
                "Correct": [tp_out],
            })

        helper.append_op(
            type="sum",
            inputs={"X": [g_total, tp_out]},
            outputs={"Out": [g_total]})
        return acc_out

    def eval(self):
        eval_program = Program()
        g_total = self._program


# This is demo for composing low level op to compute metric
class F1(Evaluator):
    def __init__(self, input, label, **kwargs):
        super(F1, self).__init__("F1", **kwargs)
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

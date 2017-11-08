from paddle.v2.framework.framework import Program, g_main_program, unique_name
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
        self._states = {}
        if kwargs.has_key("program"):
            self._program = kwargs.get("program")
        else:
            self._program = g_main_program

    def reset(self, executor, program=None):
        """
      Clear metric states at the begin of each pass/user specified batch
      """
        if program == None:
            reset_program = Program()
        else:
            reset_program = program
        block = reset_program.global_block()
        for k, var in self._states.iteritems():
            zeros = block.create_var(dtype=var.data_type)
            block.append_op(
                type="fill_constant",
                outputs={"Out": [zeros]},
                attrs={
                    "shape": var.shape,
                    "value": 0,
                })
            block.append_op(
                type="scale", inputs={"X": zeros}, outputs={"Out": var})
        executor.run(reset_program)

    def eval(self, executor, program=None):
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
        block = self._program.global_block()
        g_total = block.create_var(
            name=unique_name("Total"),
            persistable=True,
            dtype="int64",
            shape=[1])
        g_correct = helper.create_global_variable(
            name=unique_name("Correct"),
            persistable=True,
            dtype="int64",
            shape=[1])
        self._states["Total"] = g_total
        self._states["Correct"] = g_correct

        topk_out = block.create_var(dtype=input.data_type)
        topk_indices = block.create_var(dtype="int64")
        block.append_op(
            type="top_k",
            inputs={"X": [input]},
            outputs={"Out": [topk_out],
                     "Indices": [topk_indices]},
            attrs={"k": k})
        acc_out_dtype = kwargs.get("out_dtype", "float32")
        acc_out = block.create_var(dtype=acc_out_dtype)
        block.append_op(
            type="accuracy",
            inputs={
                "Out": [topk_out],
                "Indices": [topk_indices],
                "Label": [label]
            },
            outputs={
                "Accuracy": [acc_out],
                "Correct": [correct],
                "Total": [total],
            })

        block.append_op(
            type="sum",
            inputs={"X": [g_total, total]},
            outputs={"Out": [g_total]})
        block.append_op(
            type="sum",
            inputs={"X": [g_correct, correct]},
            outputs={"Out": [g_total]})
        return acc_out

    def eval(self, executor, program=None):
        if program == None:
            eval_program = Program()
        else:
            eval_program = program
        block = eval_program.global_block()
        eval_out = block.create_var(dtype=self._helper.input_dtype())
        block.append_op(
            type="elementwise_div",
            inputs={"X": self._states["Total"],
                    "Y": self._states["Correct"]},
            outputs={"Out": eval_out})
        return executor.run(eval_program, fetch_list=[eval_out])


# Demo for composing low level op to compute the F1 metric
class F1(Evaluator):
    def __init__(self, input, label, **kwargs):
        super(F1, self).__init__("F1", **kwargs)
        g_tp = helper.create_global_variable(
            name=unique_name("Tp"), persistable=True, dtype="int64", shape=[1])
        g_fp = helper.create_global_variable(
            name=unique_name("Fp"), persistable=True, dtype="int64", shape=[1])

        self._states["Tp"] = g_tp
        self._states["Fp"] = g_fp

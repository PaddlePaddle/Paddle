import numpy as np
from paddle.v2.framework.framework import Program, g_main_program, unique_name, Variable
import paddle.v2.framework.core as core


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.data_type,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


class Evaluator(object):
    """
    Evalutor Base class.

    create metric states
    add mini-batch evaluator caculate operator
    add increment operator to accumulate the metric states
    """

    def __init__(self, name, **kwargs):
        """
        init the global states
        """
        self._states = {}
        if kwargs.has_key("main_program"):
            self._main_program = kwargs.get("main_program")
        else:
            self._main_program = g_main_program

    def _update_ops(self, *args, **kwargs):
        """
        append update ops to the global states
        """
        raise NotImplementedError()

    def reset(self, executor, reset_program=None):
        """
        Clear metric states at the begin of each pass/user specified batch
        """
        if reset_program == None:
            reset_program = Program()
        else:
            reset_program = program
        block = reset_program.global_block()
        for k, var in self._states.iteritems():
            g_var = _clone_var_in_block_(block, var)
            zeros = block.create_var(dtype="float32", persistable=True)
            block.append_op(
                type="fill_constant",
                outputs={"Out": [zeros]},
                attrs={
                    "shape": g_var.shape,
                    "value": .0,
                    "data_type": 5,
                })
            block.append_op(
                type="scale", inputs={"X": zeros}, outputs={"Out": g_var})
        executor.run(reset_program, fetch_list=self._states.values())

    def eval(self, executor, eval_program=None):
        """
        Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
        """
        raise NotImplementedError()


class Accuracy(Evaluator):
    """
    Accuracy need two state variable Total, Correct
    """

    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__("accuracy", **kwargs)
        block = self._main_program.global_block()
        g_total = block.create_var(
            name=unique_name("Total"),
            persistable=True,
            dtype="int64",
            shape=[1])
        g_correct = block.create_var(
            name=unique_name("Correct"),
            persistable=True,
            dtype="int64",
            shape=[1])
        self._states["Total"] = g_total
        self._states["Correct"] = g_correct

    def _update_ops(self, input, label, k=1, **kwargs):
        block = self._main_program.global_block()
        topk_out = block.create_var(dtype=input.data_type)
        topk_indices = block.create_var(dtype="int64")
        block.append_op(
            type="top_k",
            inputs={"X": [input]},
            outputs={"Out": [topk_out],
                     "Indices": [topk_indices]},
            attrs={"k": k})
        acc_out = block.create_var(dtype=kwargs.get("out_dtype", "float32"))
        correct = block.create_var(dtype="int64", persistable=True)
        total = block.create_var(dtype="int64", persistable=True)
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
            type="cast",
            inputs={"X": [self._states["Total"]]},
            outputs={"Out": [self._states["Total"]]},
            attrs={
                "in_data_type": 5,  # float32
                "out_data_type": 2,  #int32
            })
        block.append_op(
            type="cast",
            inputs={"X": [self._states["Correct"]]},
            outputs={"Out": [self._states["Correct"]]},
            attrs={
                "in_data_type": 5,
                "out_data_type": 2,
            })

        block.append_op(
            type="elementwise_add",
            inputs={"X": [self._states["Total"]],
                    "Y": [total]},
            outputs={"Out": [self._states["Total"]]})
        block.append_op(
            type="elementwise_add",
            inputs={"X": [self._states["Correct"]],
                    "Y": [correct]},
            outputs={"Out": [self._states["Correct"]]})

        return acc_out

    def eval(self, executor, eval_program=None):
        if eval_program != None:
            eval_program = eval_program
        else:
            eval_program = Program()
        block = eval_program.global_block()
        eval_out = block.create_var(dtype=self._states["Total"].data_type)
        e_total = _clone_var_in_block_(block, self._states["Total"])
        e_correct = _clone_var_in_block_(block, self._states["Correct"])
        block.append_op(
            type="cast",
            inputs={"X": [e_total]},
            outputs={"Out": [e_total]},
            attrs={
                "in_data_type": 2,  #int32
                "out_data_type": 5,  #float32
            })
        block.append_op(
            type="cast",
            inputs={"X": [e_correct]},
            outputs={"Out": [e_correct]},
            attrs={
                "in_data_type": 2,
                "out_data_type": 5,
            })
        block.append_op(
            type="elementwise_div",
            inputs={"X": e_correct,
                    "Y": e_total},
            outputs={"Out": eval_out})
        out = executor.run(eval_program, fetch_list=[eval_out])
        return np.array(out[0])


# FIXME(dzh): add a decorator to call _update_ops automatically
def accuracy(*args, **kwargs):
    cls = Accuracy(*args, **kwargs)
    out = cls._update_ops(*args, **kwargs)
    return cls, out

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
        if kwargs.has_key("eval_program"):
            self._eval_program = kwargs.get("eval_program")
        else:
            self._eval_program = Program()

    def _update_ops(self):
        """
        append update ops to the global states
        """
        raise NotImplementedError()

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
        print reset_program
        executor.run(reset_program, fetch_list=self._states.values())

    def eval(self, executor, program=None):
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
        # block = self._eval_program.global_block()
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

        # block = self._eval_program.global_block()
        # e_correct = _clone_var_in_block_(block, correct)
        # e_total = _clone_var_in_block_(block, total)

        # block.append_op(
        #     type="sum",
        #     inputs={"X": [self._states["Total"], total]},
        #     outputs={"Out": [self._states["Total"]]})
        block.append_op(
            type="cast",
            inputs={"X": [self._states["Total"]]},
            outputs={"Out": [self._states["Total"]]},
            attrs={
                "in_data_type": 5,
                "out_data_type": 2,
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

        # g_total = self._states["Total"]
        # print g_total
        # print total

        # print "*" * 100
        # print g_total.block.program == total.block.program

        # g_total = _clone_var_in_block_(block, self._states["Total"])
        # e_total = _clone_var_in_block_(block, total)

        # block.append_op(
        #     type="sum",
        #     inputs={"X": [g_total, e_total]},
        #     outputs={"Out": [g_total]})

        # block.append_op(
        #     type="sum",
        #     inputs={"X": [self._states["Correct"], correct]},
        #     outputs={"Out": [self._states["Correct"]]})
        # print self._main_program
        return acc_out

    def eval(self, executor):
        block = self._eval_program.global_block()
        eval_out = block.create_var(dtype=self._states["Total"].data_type)
        e_correct = _clone_var_in_block_(block, correct)
        e_total = _clone_var_in_block_(block, total)
        # block.append_op(
        #     type="elementwise_div",
        #     inputs={"X": self._states["Total"],
        #             "Y": self._states["Correct"]},
        #     outputs={"Out": eval_out})
        block.append_op(
            type="elementwise_div",
            inputs={"X": e_total,
                    "Y": e_correct},
            outputs={"Out": eval_out})
        return executor.run(self._eval_program, fetch_list=[eval_out])


# Demo for composing low level ops to compute the F1 metric
class FScore(Evaluator):
    def __init__(self, input, label, beta=1.0, **kwargs):
        super(F1, self).__init__("FScore", **kwargs)
        block = self._program.global_block()
        g_tp = block.create_var(
            name=unique_name("Tp"), persistable=True, dtype="int64", shape=[1])
        g_fn = block.create_var(
            name=unique_name("Fn"), persistable=True, dtype="int64", shape=[1])
        g_fp = block.create_var(
            name=unique_name("Fp"), persistable=True, dtype="int64", shape=[1])

        self._states["Tp"] = g_tp
        self._states["Fp"] = g_fp
        self._states["Fn"] = g_fn

    def _update_ops(self):
        block = self._program.global_block()
        equal_out = block.create_var()
        block.append_op(
            type="equal",
            inputs={"X": [input],
                    "Y": [label]},
            outputs={"Out": equal_out})

        positive = block.create_var()
        block.append_op(
            type="sequence_pool",
            inputs={"X": [equal_out]},
            outputs={"Out": positive},
            attrs={"pooltype": "SUM"})
        batch = block.create_var(
            name=feed_var_name,
            type=core.VarDesc.VarType.FEED_MINIBATCH,
            persistable=True)


# def register():
accuracy = Accuracy
# def accuracy(*args, **kwargs):
#     acc = Accuracy(**kwargs)
#     return acc._update_ops(*args, **kwargs)

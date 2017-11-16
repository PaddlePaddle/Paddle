import numpy as np

import paddle.v2.fluid.layers as layers
from paddle.v2.fluid.framework import Program, unique_name, \
    Variable
from paddle.v2.fluid.layer_helper import LayerHelper

__all__ = ['Accuracy']


def _clone_var_(block, var):
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
        self.states = []
        self.metrics = []
        self.helper = LayerHelper(name, **kwargs)

    def reset(self, executor, reset_program=None):
        """
        Clear metric states at the begin of each pass/user specified batch
        """
        if reset_program is None:
            reset_program = Program()

        for var in self.states:
            assert isinstance(var, Variable)
            g_var = _clone_var_(reset_program.current_block(), var)
            layers.fill_constant(
                shape=g_var.shape,
                value=0.0,
                dtype=g_var.data_type,
                out=g_var,
                main_program=reset_program)

        executor.run(reset_program)

    def eval(self, executor, eval_program=None):
        """
        Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
        """
        raise NotImplementedError()

    def create_state(self, suffix, dtype, shape):
        state = self.helper.create_variable(
            name="_".join([unique_name(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape)
        self.states.append(state)
        return state


class Accuracy(Evaluator):
    """
    Accuracy need two state variable Total, Correct
    """

    def __init__(self, input, label, k=1, **kwargs):
        super(Accuracy, self).__init__("accuracy", **kwargs)
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.total = self.create_state(dtype='int64', shape=[1], suffix='total')
        self.correct = self.create_state(
            dtype='int64', shape=[1], suffix='correct')
        kwargs = {'main_program': main_program}
        total = self.helper.create_tmp_variable(dtype='int')
        correct = self.helper.create_tmp_variable(dtype='int')
        acc = layers.accuracy(
            input=input,
            label=label,
            k=k,
            total=total,
            correct=correct,
            **kwargs)
        total = layers.cast(x=total, data_type='int64', **kwargs)
        correct = layers.cast(x=correct, data_type='int64', **kwargs)
        layers.sums(input=[self.total, total], out=self.total, **kwargs)
        layers.sums(input=[self.correct, correct], out=self.correct, **kwargs)

        self.metrics.append(acc)

    def eval(self, executor, eval_program=None):
        if eval_program is None:
            eval_program = Program()
        block = eval_program.current_block()
        kwargs = {'main_program': eval_program}
        total = _clone_var_(block, self.total)
        correct = _clone_var_(block, self.correct)
        total = layers.cast(total, data_type='float32', **kwargs)
        correct = layers.cast(correct, data_type='float32', **kwargs)
        out = layers.elementwise_div(x=correct, y=total, **kwargs)
        return np.array(executor.run(eval_program, fetch_list=[out])[0])

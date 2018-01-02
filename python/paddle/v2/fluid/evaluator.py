import numpy as np

import layers
from framework import Program, unique_name, Variable, program_guard
from layer_helper import LayerHelper

__all__ = ['Accuracy', 'ChunkEvaluator']


def _clone_var_(block, var):
    assert isinstance(var, Variable)
    return block.create_var(
        name=var.name,
        shape=var.shape,
        dtype=var.dtype,
        type=var.type,
        lod_level=var.lod_level,
        persistable=True)


class Evaluator(object):
    """
    Base Class for all evaluators
    
    Args:
        name(str): The name of evaluator. such as, "accuracy". Used for generate 
            temporary variable name.
        main_program(Program, optional): The evaluator should be added to this 
            main_program. Default default_main_program()
        startup_program(Program, optional):The parameter should be added to this 
            startup_program. Default default_startup_program()
            
    Attributes:
        states(list): The list of state variables. states will be reset to zero 
            when `reset` is invoked.
        metrics(list): The list of metrics variables. They will be calculate 
            every mini-batch
    """

    def __init__(self, name, **kwargs):
        self.states = []
        self.metrics = []
        self.helper = LayerHelper(name, **kwargs)

    def reset(self, executor, reset_program=None):
        """
        reset metric states at the begin of each pass/user specified batch
        """
        if reset_program is None:
            reset_program = Program()

        with program_guard(main_program=reset_program):
            for var in self.states:
                assert isinstance(var, Variable)
                g_var = _clone_var_(reset_program.current_block(), var)
                layers.fill_constant(
                    shape=g_var.shape, value=0.0, dtype=g_var.dtype, out=g_var)

        executor.run(reset_program)

    def eval(self, executor, eval_program=None):
        """
        Evaluate the statistics merged by multiple mini-batches.
        """
        raise NotImplementedError()

    def create_state(self, suffix, dtype, shape):
        """
        Create state variable. 
        
        NOTE: It is not a public API.
        
        Args:
            suffix(str): the state suffix. 
            dtype(str|core.DataType): the state data type 
            shape(tuple|list): the shape of state 

        Returns: State variable

        """
        state = self.helper.create_variable(
            name="_".join([unique_name(self.helper.name), suffix]),
            persistable=True,
            dtype=dtype,
            shape=shape)
        self.states.append(state)
        return state


class Accuracy(Evaluator):
    """
    Average Accuracy for multiple mini-batches.
    """

    def __init__(self, input, label, k=1, **kwargs):
        super(Accuracy, self).__init__("accuracy", **kwargs)
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.total = self.create_state(dtype='int64', shape=[1], suffix='total')
        self.correct = self.create_state(
            dtype='int64', shape=[1], suffix='correct')
        total = self.helper.create_tmp_variable(dtype='int')
        correct = self.helper.create_tmp_variable(dtype='int')
        acc = layers.accuracy(
            input=input, label=label, k=k, total=total, correct=correct)
        total = layers.cast(x=total, dtype='int64')
        correct = layers.cast(x=correct, dtype='int64')
        layers.sums(input=[self.total, total], out=self.total)
        layers.sums(input=[self.correct, correct], out=self.correct)

        self.metrics.append(acc)

    def eval(self, executor, eval_program=None):
        if eval_program is None:
            eval_program = Program()
        block = eval_program.current_block()
        with program_guard(main_program=eval_program):
            total = _clone_var_(block, self.total)
            correct = _clone_var_(block, self.correct)
            total = layers.cast(total, dtype='float32')
            correct = layers.cast(correct, dtype='float32')
            out = layers.elementwise_div(x=correct, y=total)
        return np.array(executor.run(eval_program, fetch_list=[out])[0])


class ChunkEvaluator(Evaluator):
    """
    Accumulate counter numbers output by chunk_eval from mini-batches and 
    compute the precision recall and F1-score using the accumulated counter 
    numbers.
    """

    def __init__(
            self,
            input,
            label,
            chunk_scheme,
            num_chunk_types,
            excluded_chunk_types=None, ):
        super(ChunkEvaluator, self).__init__("chunk_eval")
        main_program = self.helper.main_program
        if main_program.current_block().idx != 0:
            raise ValueError("You can only invoke Evaluator in root block")

        self.num_infer_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_infer_chunks')
        self.num_label_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_label_chunks')
        self.num_correct_chunks = self.create_state(
            dtype='int64', shape=[1], suffix='num_correct_chunks')
        precision, recall, f1_score, num_infer_chunks, num_label_chunks, num_correct_chunks = layers.chunk_eval(
            input=input,
            label=label,
            chunk_scheme=chunk_scheme,
            num_chunk_types=num_chunk_types,
            excluded_chunk_types=excluded_chunk_types, )
        layers.sums(
            input=[self.num_infer_chunks, num_infer_chunks],
            out=self.num_infer_chunks)
        layers.sums(
            input=[self.num_label_chunks, num_label_chunks],
            out=self.num_label_chunks)
        layers.sums(
            input=[self.num_correct_chunks, num_correct_chunks],
            out=self.num_correct_chunks)

        self.metrics.extend([precision, recall, f1_score])

    def eval(self, executor, eval_program=None):
        if eval_program is None:
            eval_program = Program()
        block = eval_program.current_block()
        num_infer_chunks, num_label_chunks, num_correct_chunks = executor.run(
            eval_program,
            fetch_list=[_clone_var_(block, state) for state in self.states])
        num_infer_chunks = num_infer_chunks[0]
        num_label_chunks = num_label_chunks[0]
        num_correct_chunks = num_correct_chunks[0]
        precision = float(
            num_correct_chunks) / num_infer_chunks if num_infer_chunks else 0
        recall = float(
            num_correct_chunks) / num_label_chunks if num_label_chunks else 0
        f1_score = float(2 * precision * recall) / (
            precision + recall) if num_correct_chunks else 0
        return np.array(
            [precision], dtype='float32'), np.array(
                [recall], dtype='float32'), np.array(
                    [f1_score], dtype='float32')

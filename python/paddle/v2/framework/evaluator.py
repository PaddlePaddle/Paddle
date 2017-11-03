from paddle.v2.framework.framework import Program, g_program, g_init_program
import paddle.v2.framework.core as core


class Evaluator(object):
    """
    Evalutor Base class.

    create metric states
    add mini-batch evaluator caculate operator
    add increment operator to accumulate the metric states
    """

    def __init__(self, input=None, **kwargs):
        if "program" in kwargs:
            self._program = kwargs.get("program")
        else:
            self._program = input.program
        self._states = []

    def _create_tmp_variable(self, name, dtype):
        return self.program.current_block().create_var(
            name=unique_name(".".join([self.name, 'tmp'])),
            dtype=dtype,
            persistable=False)

    @staticmethod
    def clear(self):
        """
      clear metric states at the begin of each pass/user specified batch
      return a clear 
      """
        raise NotImplementedError()

    def evaluate(self):
        """
      Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
      """
        raise NotImplementedError()


class Accuracy(Evaluator):
    def __init__(self, input, label, k=1, **kwargs):
        super(Accuracy, self).__init__(input=input, **kwargs)
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

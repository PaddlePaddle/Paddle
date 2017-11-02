import paddle.v2.framework.op as op
import numpy as np
import paddle.v2.framework.core as core


def avg_accumulate(accumulated_var, per_eval, num_batches, place):
    t = np.array(accumulated_var.get_tensor())
    t[0] += per_eval[0]
    accumulated_var.get_tensor().set([t[0] / float(num_batches)], place)


class Evaluator(object):
    def __init__(self,
                 scope,
                 operator='accuracy',
                 input='Inference',
                 label='Label',
                 output='Output',
                 place=core.CPUPlace()):
        """
        create an evaluator for evaluating the inference.
        NOTE: default run on CPUPlace(), running on GPUPlace doesn't improve performance much.

        :param scope: the scope instance contains the input.
        :type scope: paddle.v2.framework.core.scope
        :param operator: operator name for caculating the evaluation for each mini-batch.
        :type operator: string
        :param input: output variable name of forward network.
        :type input: string
        :param label: variable name of label
        :type label: string
        """
        self.scope = scope
        self.place = place
        self.output_name = output
        self.num_batches = 0
        # create variable to store accumulated evaluator output
        eval_name = ''.join([operator, "@Eval"])
        if scope.find_var(eval_name):
            raise Exception("evaluator already exist in scope: %s" % eval_name)
        self.accumulated_var = scope.var(eval_name)
        t = self.accumulated_var.get_tensor()
        t.set_dims((1, ))
        t.set([0.0], place)
        # self.accumulated_var = block.create_var(block, name=eval_name, shape=(1,))
        # self.accumulated_var.get_tensor().set([0.0])
        # create operator of evaluation
        var_map = dict()  # var name -> variable
        var_map[input] = [input]
        var_map[label] = [label]
        var_map[output] = [output]
        self.op = op.Operator(operator, **var_map)

    def evaluate(self, ctx, accumulator=avg_accumulate):
        self.op.run(self.scope, ctx)
        per_eval = np.array(self.scope.find_var(self.output_name).get_tensor())
        self.num_batches += 1
        accumulator(self.accumulated_var, per_eval, self.num_batches,
                    self.place)

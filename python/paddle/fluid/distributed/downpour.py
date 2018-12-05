from .node import DownpourServer
from .node import DownpourWorker
from ..backward import append_backward
import ps_pb2 as pslib
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format

class DownpourSGD(object):
    """
    Distributed optimizer of downpour stochastic gradient descent
    Standard implementation of Google's Downpour SGD
    in Large Scale Distributed Deep Networks

    Args:
        learning_rate (float): the learning rate used to update parameters. \
        Can be a float value
    Examples:
        .. code-block:: python
    
             downpour_sgd = fluid.distributed.DownpourSGD(learning_rate=0.2)
             downpour_sgd.minimize(cost)
    """
    def __init__(self, learning_rate=0.001, window=1):
        # todo(guru4elephant): add more optimizers here as argument
        # todo(guru4elephant): make learning_rate as a variable
        self.learning_rate_ = learning_rate
        self.window_ = window
        self.type = "downpour"
    
    def minimize(self, loss, startup_program=None,
                 parameter_list=None, no_grad_set=None):
        params_grads = sorted(append_backward(
            loss, parameter_list, no_grad_set), key=lambda x:x[0].name)
        table_name = find_distributed_lookup_table(loss.block.program)
        prefetch_slots = find_distributed_lookup_table_inputs(
            loss.block.program, table_name)
        prefetch_slots_emb = find_distributed_lookup_table_outputs(
            loss.block.program, table_name)
        server = DownpourServer()
        # window is communication strategy
        worker = DownpourWorker(self.window_)
        # Todo(guru4elephant): support multiple tables definitions
        # currently support one big sparse table
        sparse_table_index = 0
        # currently merge all dense parameters into one dense table
        dense_table_index = 1
        server.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        server.add_dense_table(dense_table_index, self.learning_rate_, 
                               params_grads[0], params_grads[1])
        worker.add_sparse_table(sparse_table_index, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        worker.add_dense_table(dense_table_index, self.learning_rate_, 
                               params_grads[0], params_grads[1])
        ps_param = pslib.PSParameter()
        ps_param.server_param.CopyFrom(server.get_desc())
        ps_param.trainer_param.CopyFrom(worker.get_desc())
        # Todo(guru4elephant): figure out how to support more sparse parameters
        # currently only support lookup_table
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        ps_param_str = text_format.MessageToString(ps_param)
        return [ps_param_str, worker_skipped_ops]

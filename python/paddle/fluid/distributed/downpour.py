from .node import DownpourServer
from .node import DownpourWorker
from ..backward import append_backward
import ps_pb2 as pslib
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_inputs
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table_outputs
from google.protobuf import text_format

class DownpourSGD(object):
    def __init__(self, learning_rate=0.001, window=1):
        # todo(guru4elephant): if optimizer is not None, will warning here
        self.learning_rate_ = learning_rate
        self.window_ = window

    def minimize(self, loss, startup_program=None,
                 parameter_list=None, no_grad_set=None):
        params_grads = sorted(append_backward(loss), key=lambda x:x[0].name)
        table_name = find_distributed_lookup_table(loss.block.program)
        prefetch_slots = find_distributed_lookup_table_inputs(
            loss.block.program, table_name)
        prefetch_slots_emb = find_distributed_lookup_table_outputs(
            loss.block.program, table_name)
        server = DownpourServer()
        worker = DownpourWorker(self.window_)
        server.add_sparse_table(0, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        server.add_dense_table(1, self.learning_rate_, params_grads[0], params_grads[1])
        worker.add_sparse_table(0, self.learning_rate_,
                                prefetch_slots, prefetch_slots_emb)
        worker.add_dense_table(1, self.learning_rate_, params_grads[0], params_grads[1])
        ps_param = pslib.PSParameter()
        ps_param.server_param.CopyFrom(server.get_desc())
        #ps_param.worker_param.CopyFrom(worker.get_desc())
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        ps_param_str = text_format.MessageToString(ps_param)
        return [ps_param_str, worker_skipped_ops, text_format.MessageToString(worker.get_desc())]

import paddle.fluid as fluid
import pslib_pb2 as pslib
from .node import DownpourServer
from .node import DownpourWorker
from paddle.fluid.distribute_lookup_table import find_distributed_lookup_table

class DownpourSGD(object):
    def __init__(self, optimizer=opt, learning_rate=0.001, window=1):
        # todo(guru4elephant): if optimizer is not None, will warning here
        self.learning_rate_ = opt.learning_rate
        self.window_ = window

    def minimize(self, loss, startup_program=None, 
                 parameter_list=None, no_grad_set=None, 
                 prefetch_slots=None, prefetch_slots_emb=None):
        params_grads = sorted(append_backward(loss), key=lambda x:x[0].name)
        table_name = fluid_distributed_lookup_table(loss.block.program)
        server = DownpourServer()
        worker = DownpourWorker()
        server.add_sparse_table(0, learning_rate, 
                                prefetch_slots, prefetch_slots_emb)
        server.add_dense_table(1, learning_rate, params, grads)
        worker.add_sparse_table(0, learning_rate, 
                                prefetch_slots, prefetch_slots_emb)
        worker.add_dense_table(1, learning_rate, params, grads)

        ps_param = pslib.PSParameter()
        ps_param.server_param.CopyFrom(server.get_desc())
        ps_param.worker_param.CopyFrom(worker.get_desc())
        worker_skipped_ops = ["lookup_table", "lookup_table_grad"]
        
        return [solver_desc, parallel_desc]
        
            

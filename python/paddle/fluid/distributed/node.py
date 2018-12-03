import ps_pb2 as pslib

class Server(object):
    def __init__(self):
        pass


class Worker(object):
    def __init__(self):
        pass


class DownpourServer(Server):
    def __init__(self):
        #self.server_ = pslib.ServerParameter().downpour_server_param
        self.server_ = pslib.ServerParameter()

    def add_sparse_table(self, table_id, learning_rate,
                         slot_key, slot_value_var, slot_grad_var):
        #table = self.server_.downpour_table_param.add()
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.type = PS_SPARSE_TABLE
        table.accessor.accessor_class = "DownpourFeatureValueAccessor"
        table.accessor.dense_sgd_param.adam.learning_rate = learning_rate
        table.accessor.fea_dim = slot_value_var[0].shape[1]

    def add_dense_table(self, table_id, learning_rate, 
                        param_var, grad_var):
        #table = self.server_.downpour_table_param.add()
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.type = PS_DENSE_TABLE
        table.accessor.accessor_class = "DownpourDenseValueAccessor"
        table.accessor.sparse_sgd_param.learning_rate = learning_rate
        table.accessor.fea_dim = 1
        #table.accessor.fea_dim = reduce(lambda x, y: x.shape, 1 for x in param_var)

    def get_desc(self):
        return self.server_


class DownpourWorker(Worker):
    def __init__(self, window):
        self.window = window
        #self.worker_ = pslib.WorkerParameter().downpour_worker_param
        #self.worker_ = pslib.WorkerParameter()
        self.worker_ = pslib.DownpourTrainerParameter()
        #self.worker_.pull_dense_per_batch = window
        #self.worker_.push_dense_per_batch = window
        #self.worker_.downpour_worker_param.pull_dense_per_batch = window
        #self.worker_.downpour_worker_param.push_dense_per_batch = window
        self.worker_.pull_dense_per_batch = window
        self.worker_.push_dense_per_batch = window
        print(self.worker_)

    def add_sparse_table(self, table_id, 
                         slot_keys, slot_value_vars, slot_grad_vars):
        #table = self.worker_.sparse_table.add()
        table = self.worker_.downpour_worker_param.sparse_table.add()
        table.table_id = table_id
        table.slot.extend(slot_keys)
        self.worker_.extend([grad.name for grad in slot_grad_vars])

    def add_dense_table(self, table_id, param_vars, grad_vars):
        #table = self.worker_.dense_table.add()
        table = self.worker_.downpour_worker_param.dense_table.add()
        table.table_id = table_id
        table.dense_variable_name.extend([p.name for p in param_vars])
        table.dense_gradient_variable_name.extend([g.name for g in grad_vars])

    def get_desc(self):
        return self.worker_

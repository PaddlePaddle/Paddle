import ps_pb2 as pslib

class Server(object):
    def __init__(self):
        pass


class Worker(object):
    def __init__(self):
        pass


class DownpourServer(Server):
    def __init__(self):
        self.server_ = pslib.ServerParameter()

    def add_sparse_table(self, table_id, learning_rate,
                         slot_key_vars, slot_value_var):
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.type = pslib.PS_SPARSE_TABLE
        table.accessor.accessor_class = "DownpourFeatureValueAccessor"
        table.accessor.dense_sgd_param.adam.learning_rate = learning_rate
        table.accessor.fea_dim = abs(reduce(lambda x, y: x * y, 
                                            slot_value_var[0].shape, 1))

    def add_dense_table(self, table_id, learning_rate, 
                        param_var, grad_var):
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.type = pslib.PS_DENSE_TABLE
        table.accessor.accessor_class = "DownpourDenseValueAccessor"
        table.accessor.sparse_sgd_param.learning_rate = learning_rate
        fea_dim = 0
        for param in param_var:
            fea_dim += reduce(lambda x, y: x * y, param.shape, 1)
        table.accessor.fea_dim = fea_dim

    def get_desc(self):
        return self.server_


class DownpourWorker(Worker):
    def __init__(self, window):
        self.window = window
        self.worker_ = pslib.DownpourTrainerParameter()
        self.worker_.pull_dense_per_batch = window
        self.worker_.push_dense_per_batch = window

    def add_sparse_table(self, table_id, learning_rate,
                         slot_key_vars, slot_value_vars):
        table = self.worker_.sparse_table.add()
        table.table_id = table_id
        table.slot_key.extend(
            [var.name for var in slot_key_vars])
        table.slot_value.extend(
            [var.name for var in slot_value_vars])
        table.slot_gradient.extend(
            [var.name + "@GRAD" for var in slot_value_vars])

    def add_dense_table(self, table_id, learning_rate, 
                        param_vars, grad_vars):
        table = self.worker_.dense_table.add()
        table.table_id = table_id
        table.dense_variable_name.extend([p.name for p in param_vars])
        table.dense_gradient_variable_name.extend([g.name for g in grad_vars])

    def get_desc(self):
        return self.worker_

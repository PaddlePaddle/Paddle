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
        self.server_.downpour_server_param.service_param.start_server_port = 0
        self.server_.downpour_server_param.service_param.server_class = "DownpourBrpcPsServer"
        self.server_.downpour_server_param.service_param.client_class = "DownpourBrpcPsClient"
        self.server_.downpour_server_param.service_param.service_class = "DownpourPsService"
        self.server_.downpour_server_param.service_param.start_server_port = 0
        self.server_.downpour_server_param.service_param.server_thread_num = 12

    def add_sparse_table(self, table_id, learning_rate,
                         slot_key_vars, slot_value_var):
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.table_class = "DownpourSparseTable"
        table.type = pslib.PS_SPARSE_TABLE
        table.accessor.accessor_class = "DownpourFeatureValueAccessor"
        table.accessor.sparse_sgd_param.learning_rate = learning_rate
        table.accessor.sparse_sgd_param.initial_g2sum = 3
        table.accessor.sparse_sgd_param.initial_range = 1e-4
        table.accessor.sparse_sgd_param.weight_bounds.extend([-10, 10])
        
        table.accessor.embedx_dim = 8
        table.accessor.embedx_threshold = 5
        table.accessor.fea_dim = 11 
        #table.accessor.fea_dim = abs(reduce(lambda x, y: x * y, 
        #                                    slot_value_var[0].shape, 1))
        table.accessor.downpour_accessor_param.nonclk_coeff = 0.1
        table.accessor.downpour_accessor_param.click_coeff = 2
        table.accessor.downpour_accessor_param.base_threshold = 0.2
        table.accessor.downpour_accessor_param.delta_threshold = 0.15
        table.accessor.downpour_accessor_param.delta_keep_days = 31
        table.accessor.downpour_accessor_param.show_click_decay_rate = 0.999
        table.accessor.downpour_accessor_param.delete_threshold = 0.8

    def add_dense_table(self, table_id, learning_rate, 
                        param_var, grad_var):
        table = self.server_.downpour_server_param.downpour_table_param.add()
        table.table_id = table_id
        table.table_class = "DownpourDenseTable"
        table.type = pslib.PS_DENSE_TABLE
        table.accessor.accessor_class = "DownpourDenseValueAccessor"
        table.accessor.dense_sgd_param.name = "adam" 
        table.accessor.dense_sgd_param.adam.learning_rate = learning_rate
        table.accessor.dense_sgd_param.adam.avg_decay_rate = 0.999993 
        table.accessor.dense_sgd_param.adam.ada_decay_rate = 0.9999 
        table.accessor.dense_sgd_param.adam.ada_epsilon = 1e-8
        table.accessor.dense_sgd_param.adam.mom_decay_rate = 0.99
        table.accessor.dense_sgd_param.naive.learning_rate = 0.0002
        fea_dim = 0
        for param in filter(lambda x: x.name.find("embedding") == -1, param_var):
            fea_dim += reduce(lambda x, y: x * y, param.shape, 1)
        table.accessor.fea_dim = fea_dim

    def get_desc(self):
        return self.server_


class DownpourWorker(Worker):
    def __init__(self, window):
        self.window = window
        self.worker_ = pslib.DownpourTrainerParameter()
        #self.worker_.pull_dense_per_batch = window
        #self.worker_.push_dense_per_batch = window

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
        table.dense_variable_name.extend(filter(lambda x: x.find("embedding") == -1, [p.name for p in param_vars]))
        table.dense_gradient_variable_name.extend(filter(lambda x: x.find("embedding") == -1, [g.name for g in grad_vars]))

    def get_desc(self):
        return self.worker_

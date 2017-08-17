import paddle.v2.framework.network as network
import paddle.v2.framework.core as core

__all__ = ['Model', 'g_model', 'ParameterAttribute']


class ParameterAttribute(object):
    def __init__(self,
                 name=None,
                 initial_max=None,
                 initial_min=None,
                 initial_mean=None,
                 initial_std=None,
                 initial_seed=0):
        self.name = name

        if initial_min is None and initial_max is None and initial_mean is None and initial_std is None:
            initial_min = -1.0
            initial_max = 1.0

        if initial_max is not None and initial_min is not None:
            self.init_strategy = ('uniform_random', {
                'min': initial_min,
                'max': initial_max,
                'seed': initial_seed
            })
        elif initial_mean is not None and initial_std is not None:
            self.init_strategy = ('gauss_random', {
                'mean': initial_mean,
                'std': initial_std,
                'seed': initial_seed
            })
        else:
            raise ValueError()

    @staticmethod
    def default_weight_attr():
        return ParameterAttribute()

    @staticmethod
    def default_bias_attr():
        # TODO(yy): Change it to FillZero.
        return ParameterAttribute(initial_min=-0.0001, initial_max=0.0001)


class Model(object):
    def __init__(self, place=None):
        self.init_network = network.Network()
        self.network = network.Network()

        if place is None:
            self.device_context = None
            self.place = None
        else:
            self.device_context = core.DeviceContext.create(place)
            self.place = None

        self.global_scope = core.Scope()
        self.cur_scope = self.global_scope
        self.name_counter = 0
        self.all_param_names = set()
        self.has_been_run = False

    def next_name(self, prefix):
        name = prefix + str(self.name_counter)
        self.name_counter += 1
        return name

    def create_parameter(self, name_prefix, param_attr, dims):
        if not isinstance(param_attr, ParameterAttribute):
            raise TypeError()
        if param_attr.name is None:
            param_attr.name = self.next_name(name_prefix)

        if self.cur_scope.find_var(param_attr.name) is not None:
            raise ValueError("Parameter {} has been created before",
                             param_attr.name)

        self.cur_scope.new_var(param_attr.name).get_tensor()

        op_type, attrs = param_attr.init_strategy
        attrs['dims'] = dims
        attrs['Out'] = param_attr.name

        op_func = getattr(self.init_network, op_type)
        pname = op_func(**attrs)
        self.all_param_names.add(pname)

        self.init_network.infer_shape(
            len(self.init_network) - 1, self.cur_scope)

        return pname

    def add_op_and_infer_shape(self, op_type, **kwargs):
        out = self.network.create_and_add_op(op_type, **kwargs)
        return_value = out
        if out is None:
            return

        if isinstance(out, unicode) or isinstance(out, str):
            out = [out]

        for o in out:
            v = self.cur_scope.find_var(o)
            if v is None:
                v = self.cur_scope.new_var(o)
            v.get_tensor()

        op_idx = len(self.network) - 1
        self.network.infer_shape(op_idx, self.cur_scope)

        return return_value

    def set_place(self, place):
        if self.has_been_run:
            raise ValueError("Cannot set place to model after run")

        self.device_context = core.DeviceContext.create(place)
        self.place = place

    def init_parameters(self):
        if self.device_context is None:
            # TODO(yy): Log warning here
            self.set_place(core.CPUPlace())
        if not self.has_been_run:
            self.has_been_run = True

        self.init_network.run(self.global_scope, self.device_context)

    def run(self):
        if not self.has_been_run:
            self.has_been_run = True
        self.network.infer_shape(self.global_scope)
        self.network.run(self.global_scope, self.device_context)

    def feed_data(self, data):
        for key in data:
            tensor = self.global_scope.find_var(key).get_tensor()
            d = data[key]
            tensor.set_dims(d.shape)
            tensor.set(d, self.place)

    def find_tensor(self, var_name):
        return self.global_scope.find_var(var_name).get_tensor()


g_model = Model()

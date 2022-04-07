#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from collections import OrderedDict
import paddle

COMM_OP_TYPE = [
    "send_v2", "recv_v2", "c_broadcast", "c_allgather", "c_allreduce_sum"
]
NON_COMP_TYPE = ["while"] + COMM_OP_TYPE
OP_COST_FACTORY = {}


def _parse_op_to_desc(op, dist_context=None):
    desc = {}
    desc["op"] = op.type
    vars = op.block.vars
    input_desc = OrderedDict()
    for input_name in op.input_names:
        var_name_list = op.input(input_name)
        var_desc = []
        for var_name in var_name_list:
            var = vars[var_name]
            shape = None
            if dist_context is not None:
                dist_tensor = dist_context.get_dist_tensor_for_program(var)
                shape = dist_tensor.local_sizes()
            else:
                shape = var.shape
            assert shape is not None
            var_desc.append((var.dtype, shape))
        input_desc[input_name] = var_desc
    desc["inputs"] = input_desc

    output_desc = OrderedDict()
    for out_name in op.output_names:
        var_name_list = op.output(out_name)
        var_desc = []
        for var_name in var_name_list:
            var = vars[var_name]
            shape = None
            if dist_context is not None:
                dist_tensor = dist_context.get_dist_tensor_for_program(var)
                shape = dist_tensor.local_sizes()
            else:
                shape = var.shape
            assert shape is not None
            var_desc.append((var.dtype, shape))
        output_desc[out_name] = var_desc
    desc["outputs"] = output_desc

    attr_desc = op.all_attrs
    desc["attrs"] = attr_desc

    return desc


def parse_to_desc(op=None, dist_op=None, dist_context=None):
    desc = None
    if op is None and dist_op is not None and dist_context is not None:
        desc = _parse_op_to_desc(
            op=dist_op.serial_op, dist_context=dist_context)
    elif op is not None and dist_op is None and dist_context is None:
        desc = _parse_op_to_desc(op)

    return desc


def parse_desc_to_str(desc):
    def _parse_dtype(dtype):
        dtype_str = ""
        if dtype == paddle.float32:
            dtype_str = "float32"
        elif dtype == paddle.float16:
            dtype_str = "float16"
        elif dtype == paddle.int32:
            dtype_str = "int32"
        elif dtype == paddle.int64:
            dtype_str = "int64"
        elif dtype == paddle.unit8:
            dtype_str = "unit8"
        else:
            raise TypeError("Unsupported dtype {}".format(dtype))
        return dtype_str

    assert isinstance(desc, dict)
    desc_str_list = []
    desc_str = None
    dtype_str_list = []
    dims_list = []
    shape_list = []

    desc_str_list.append(desc["op"])
    inputs = desc["inputs"]
    for key, item in inputs.items():
        for dtype, shape in item:
            dtype_str_list.append(_parse_dtype(dtype))
            shape_list += list(shape)
            dims = len(shape)
            dims_list.append(dims)

    dtype_str = "*".join(dtype_str_list)
    dims_list = [str(item) for item in dims_list]
    dims_str = "*".join(dims_list)

    shape_list = [str(item) for item in shape_list]
    shape_str = "[" + ",".join(shape_list) + "]"
    desc_str_list += [dtype_str, dims_str, shape_str]
    desc_str = "_".join(desc_str_list)

    return desc_str


class CommContext:
    _instance = None
    _has_instance = False

    def __init__(self, cluster):
        if CommContext._has_instance:
            return
        self.cluster = cluster
        self._alpha_base_ring = 8.4
        self._alpha_base_tree = 0
        self._alpha_inter = None
        self._alpha_intra
        self._beta = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            _has_instance = True
        return cls._instance

    @property
    def alpha_inter(self):
        if self._alpha_inter is None:
            if cluster.alpha.inter == "NVL":
                self._alpha_inter = 3.4
            elif cluster.alpha.inter == "PHB":
                self._alpha_inter = 5.7
        return self._alpha_inter

    @property
    def alpha_intra(self):
        if self._alpha_intra is None:
            if cluster.alpha.intra == "NVL":
                self._alpha_intra = 28
            elif cluster.alpha.intra == "PHB":
                self._alpha_intra = 28
        return self._alpha_intra

    @property
    def alpha_base_ring(self):
        return self._alpha_base_ring

    @property
    def alpha_base_tree(self):
        return self._alpha_base_tree

    def get_beta(self, ranks):
        key = ','.join(map(str, sorted(ranks)))
        max_beta = None
        if key in self._beta.keys:
            max_beta = self._beta[key]
        else:
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    if min_beta == None:
                        min_beta = cluster.get_beta(ranks[i], ranks[j])
                    else:
                        beta = cluster.get_beta(ranks[i], ranks[j])
                        if beta > max_beta:
                            max_beta = beta
            self._beta[key] = max_beta

        return max_beta


class Cost:
    def __init__(self, time=0, memory=0, flops=0):
        self.time = time
        self.memory = memory
        self.flops = flops

    def _check_time(self, val):
        assert val >= 0, "Time must be greater than or equal to 0."

    def _check_memory(self, val):
        assert isinstance(
            val, int) and val >= 0, "Memory must be int and greater than 0."

    def _check_flops(self, val):
        assert isinstance(
            val, int) and val >= 0, "FLOPs must be int and greater than 0."

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._check_time(val)
        self._time = val

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, val):
        self._check_memory(val)
        self._memory = val

    @property
    def flops(self):
        return self._flops

    @flops.setter
    def flops(self, val):
        self._check_flops(val)
        self._flops = val

    def __add__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time + rhs.time
        memory = self.memory + rhs.memory
        flops = self.flops + rhs.flops
        assert (time >= 0 and memory >= 0 and flops >= 0)
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time - rhs.time
        memory = self.memory - rhs.memory
        flops = self.flops - rhs.flops
        assert (time >= 0 and memory >= 0 and flops >= 0)
        return Cost(time, memory, flops)


class OpCost:
    def __init__(self, op=None, op_desc=None):
        assert (op is not None and op_desc is None) or (op is None and
                                                        op_desc is not None)
        self._op = op
        self._op_desc = op_desc
        self._cost = self.calc_cost()

    @property
    def op(self):
        return self._op

    @property
    def op_desc(self):
        return self._op_desc

    @property
    def cost(self):
        return self._cost

    def calc_time(self):
        return 0

    def calc_memory(self):
        return 0

    def calc_flops(self):
        return 0

    def calc_cost(self):
        time = self.calc_time()
        memory = self.calc_memory()
        flops = self.calc_flops()
        cost = Cost(time, memory, flops)
        return cost


class CommOpCost(OpCost):
    OP_TYPE = "COMM"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super(CommOpCost, self).__init__(op=op, op_desc=op_desc)
        self._check_comm_op_type()
        self._comm_context = comm_context

    @property
    def comm_context(self):
        return self._comm_context

    @classmethod
    def _check_comm_op_type(cls):
        if cls.OP_TYPE != "COMM":
            if cls.OP_TYPE not in COMM_OP_TYPE:
                raise TypeError("Please Check op type in {}, but got {}.".
                                format(COMM_OP_TYPE, cls.OP_TYPE))


class CompOpCost(OpCost):
    OP_TYPE = "COMP"

    def __init__(self, op=None, op_desc=None, cluster=None):
        super(CompOpCost, self).__init__(op=op, op_desc=op_desc)
        self._check_comp_op_type()
        self.cluster = cluster

    @classmethod
    def _check_comp_op_type(cls):
        if cls.OP_TYPE != "COMP":
            if cls.OP_TYPE in NON_COMP_TYPE:
                raise TypeError("Please Check op type not in {}, but got {}.".
                                format(NON_COMP_TYPE, cls.OP_TYPE))


def register_op_cost(cls):
    op_type = cls.OP_TYPE

    def register(op_type):
        OP_COST_FACTORY[op_type] = cls

    return register(op_type)


def calc_time_from_model(op=None, desc=None, cluster=None, comm_context=None):
    op_type = op.type if op is not None else desc["op"]
    if op_type in COMM_OP_TYPE:
        op_cost = OP_COST_FACTORY[op_type](op=op,
                                           op_desc=desc,
                                           comm_context=comm_context)
    elif op_type not in NON_COMP_TYPE:
        op_cost = OP_COST_FACTORY[op_type](op=op, op_desc=desc, cluster=cluster)
    time = op_cost.calc_time()
    return time

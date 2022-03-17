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

import paddle


class Cost:
    def __init__(self, time=0., memory=0, flops=0):
        self.time = time
        self.memory = memory
        self.flops = flops

    def _check_time(self, val):
        assert isinstance(
            val, float
        ) and val >= 0, "Time must be float and greater than or equal to 0."

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
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time - rhs.time
        memory = self.memory - rhs.memory
        flops = self.flops - rhs.flops
        return Cost(time, memory, flops)


class OpCost:
    def __init__(self, op=None, op_info=None, dist_context=None, cluster=None):
        assert not (op is not None and op_info is not None)
        assert not (op is None and op_info is None)
        self._op = op
        self._op_info = op_info
        self._dist_context = dist_context
        self._cluster = cluster
        self._cost = None

    @property
    def op(self):
        return self._op

    @property
    def op_info(self):
        return self._op_info

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
        raise NotImplementedError


class CommContext:
    _instance = None
    _has_instance = False

    def __init__(self, cluster):
        if CommContext._has_instance:
            return
        self._alpha, self._beta = self.init_comm_args(cluster)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            _has_instance = True
        return cls._instance

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def init_comm_args(self, cluster):
        alpha = 0
        beta = {}
        return aplha, beta


class CommOpCost(OpCost):
    COMM_OP_TYPE = [
        "send_v2", "recv_v2", "c_broadcast", "c_allgather", "c_allreduce_sum"
    ]
    OP_TYPE = "CommOp"

    def __init__(self, op=None, op_info=None, dist_context=None, cluster=None):
        super(CommOpCost, self).__init__(op, op_info, dist_context, cluster)
        self._check_op_type(op, op_info)
        self._comm_context = CommContext(self.cluster)

    @property
    def comm_context(self):
        return self._comm_context

    def _check_comm_op_type(op, op_info):
        op_type = None
        if op is not None:
            op_type = op.type
        elif op_info is not None:
            op_type = op_info["op"]

        if op_type not in COMM_OP_TYPE:
            raise TypeError("Please Check op type in {}, but got {}.".format(
                COMM_OP_TYPE, op_type))

        if cls.OP_TYPE != "CommOp":
            assert op_type == cls.OP_TYPE

    def calc_cost(self):
        # For comm op, its memory cost is 0 and flops is 0 in default
        cost = Cost()
        return cost


def parse_op_info(op=None, op_info=None):
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

    op_info_str_list = []
    op_info_str = None
    dtype_str_list = []
    dims_list = []
    shape_list = []
    if op is not None:
        op_info_str_list.append(str(op.type))
        vars = op.block.vars
        for input_name in op.input_names:
            var_names = op.input(input_name)
            for var_name in var_names:
                var = vars[var_name]
                dtype = _parse_dtype(var.dtype)
                dtype_str_list.append(dtype)
                shape_list += list(var.shape)
                dims = len(shape)
                dims_list.append(dims)

    elif op_info is not None:
        op_info_str_list.append(op_info["op"])
        input_list = op_info["input"]
        for input_info in input_list:
            for dtype, shape in input_info:
                dtype_str_list.append(_parse_dtype(dtype))
                shape_list += list(shape)
                dims = len(shape)
                dims_list.append(dims)

        dtype_str = "*".join(dtype_str_list)
        dims_str = "*".join(dims_list)
        shape_str = "[" + ",".join(shape_list) + "]"
        op_info_str_list.append(dtype_str, dims_str, shape_str)
        op_info_str = "_".join(op_info_str_list)

    return op_info_str


def calc_time_from_benchmark(op_info):
    return 0


class CompOpCost(OpCost):
    SPEC_OP_TYPE = ["while"] + CommOpCost.COMM_OP_TYPE
    OPTYPE = "CompOp"

    def __init__(self, op=None, op_info=None, dist_context=None, cluster=None):
        super(CompOpCost, self).__init__(op, op_info, dist_context, cluster)
        self._check_comp_op_type(op, op_info)
        self._cost = self.calc_cost()

    def _check_comp_op_type(self, op, op_info):
        op_type = None
        if op is not None:
            op_type = op.type
        elif op_info is not None:
            op_type = op_info["op"]

        if op_type in SPEC_OP_TYPE:
            raise TypeError("Please Check op type not in {}, but got {}.".
                            format(SPEC_OP_TYPE, op_type))

        if cls.OP_TYPE != "CompOp":
            assert op_type == cls.OP_TYPE

    def calc_flops(self):
        return 0

    def calc_time(self):
        info = parse_op_info(self.op, self.op_info)
        time = calc_time_from_benchmark(info)
        return time

    def calc_memory(self):
        return 0

    def calc_cost(self):
        time = self.calc_time()
        memory = self.calc_memory()
        flops = self.calc_flops()
        cost = Cost(time, memory, flops)
        return cost


OP_COST_FACTORY = {}


def register_op_cost(cls):
    op_type = cls.OP_TYPE

    def register(op_type):
        OP_COST_FACTORY[op_type] = cls

    return register(op_type)

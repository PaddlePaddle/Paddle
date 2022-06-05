# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from functools import reduce

import paddle

from ..cluster import LinkType
from ..process_group import get_process_group

COMM_OP_TYPE = [
    "send_v2", "recv_v2", "c_broadcast", "c_allgather", "c_allreduce_sum",
    "c_identity"
]
NON_COMP_TYPE = ["while"] + COMM_OP_TYPE
_g_op_cost_factory = {}


def build_comm_desc(op_type, group_ranks, dtype, shape, attrs=None):
    desc = {}
    desc["op"] = op_type
    desc["group_ranks"] = group_ranks
    desc["inputs"] = {"X": [(dtype, shape)]}
    if attrs is not None:
        desc["attrs"] = attrs
    return desc


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
        desc = _parse_op_to_desc(op=dist_op.serial_op,
                                 dist_context=dist_context)
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

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            _has_instance = True
        return cls._instance

    def __init__(self, cluster):
        if CommContext._has_instance:
            return
        self.beta = {}
        self.hops = {}
        assert cluster is not None
        self.cluster = cluster
        # if cluster has no info about those vars, it will be set by default
        self.base_ring = None
        self.base_tree = None
        # self.base_inter_ring = None
        # self.base_inter_tree = None
        self.intra_ring = None
        self.intra_tree = None
        self.inter_ring = None
        self.inter_tree = None
        self.switch = None
        self._post_init()

    def _post_init(self):
        alpha_latency = self.cluster.alpha_latency
        if alpha_latency is None:
            # set default
            self.base_ring = 8.4
            self.base_tree = 0.
            # NVL in default
            self.intra_ring = 3.4
            self.intra_tree = 28
            # NET in default
            self.inter_ring = 9.6
            self.inter_tree = 28
            self.switch = 10.0
        else:
            base_ring = alpha_latency.base_ring
            self.base_ring = base_ring if base_ring is not None else 8.4

            base_tree = alpha_latency.base_tree
            self.base_tree = base_tree if base_tree is not None else 0.

            intra_ring = alpha_latency.intra_ring
            if intra_ring == LinkType.NVL:
                self.intra_ring = 3.4
            elif intra_ring == LinkType.PHB:
                self.intra_ring = 5.7
            elif intra_ring is not None:
                self.intra_ring = intra_ring
            else:
                # NVL Default
                self.intra_ring = 3.4

            intra_tree = alpha_latency.intra_tree
            if intra_tree == LinkType.NVL:
                self.intra_tree = 28
            elif intra_tree == LinkType.PHB:
                self.intra_tree = 28
            elif intra_tree is not None:
                self.intra_tree = intra_tree
            else:
                # NVL Default
                self.intra_tree = 28

            inter_ring = alpha_latency.inter_ring
            if inter_ring == LinkType.NET:
                self.inter_ring = 9.6
            elif inter_ring is not None:
                self.inter_ring = inter_ring
            else:
                # NET Default
                self.inter_ring = 9.6

            inter_tree = alpha_latency.inter_tree
            if inter_tree == LinkType.NET:
                self.inter_tree = 28
            elif inter_tree is not None:
                self.inter_tree = inter_tree
            else:
                # NET Default
                self.inter_tree = 28

            switch = alpha_latency.switch
            self.switch = switch if switch is not None else 10

            assert self.base_ring is not None
            assert self.base_tree is not None
            assert self.intra_ring is not None
            assert self.intra_tree is not None
            assert self.inter_ring is not None
            assert self.inter_tree is not None
            assert self.switch is not None

    def get_max_beta(self, ranks):
        # NOTE: Get beta by ring, even in the case of tree such as tree broadcast
        ranks = self.cluster.convert_rank_to_device_id(ranks)
        key = ','.join(map(str, sorted(ranks)))
        max_beta = None
        if key in self.beta:
            max_beta = self.beta[key]
        else:
            for i in range(len(ranks)):
                for j in range(i + 1, len(ranks)):
                    forward_order_beta = self.cluster.get_beta(
                        ranks[i], ranks[j])
                    backward_order_beta = self.cluster.get_beta(
                        ranks[j], ranks[i])
                    beta = forward_order_beta if forward_order_beta > backward_order_beta else backward_order_beta
                    if max_beta == None:
                        max_beta = beta
                    else:
                        if beta > max_beta:
                            max_beta = beta
            self.beta[key] = max_beta

        return max_beta

    def get_hops(self, ranks):
        key = ','.join(map(str, sorted(ranks)))
        hops = 0
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                hop = self.cluster.get_hop(ranks[i], ranks[j])
                hops += hop
        self.hops[key] = hops

        return hops


class Cost:

    def __init__(self, time=0, memory=0, flops=0):
        self.time = time
        self.memory = memory
        self.flops = flops

    def _check_time(self, val):
        assert val >= 0, "Time must be greater than or equal to 0."

    def _check_memory(self, val):
        assert isinstance(
            val,
            int) and val >= 0, "Memory must be int and greater than equal to 0."

    def _check_flops(self, val):
        assert isinstance(
            val,
            int) and val >= 0, "FLOPs must be int and greater than equal to 0."

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
        self._op = op
        self._op_desc = op_desc
        self._cost = None

    @property
    def op(self):
        return self._op

    @property
    def op_desc(self):
        return self._op_desc

    @property
    def time(self):
        return self.cost.time

    @property
    def memory(self):
        return self.cost.memory

    @property
    def flops(self):
        return self.cost.flops

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

    def __add__(self, rhs):
        assert isinstance(rhs, (OpCost, Cost))
        time = 0
        memory = 0
        flops = 0
        if isinstance(rhs, OpCost):
            time = self.cost.time + rhs.cost.time
            memory = self.cost.memory + rhs.cost.memory
            flops = self.cost.flops + rhs.cost.flops
            assert (time >= 0 and memory >= 0 and flops >= 0)
        elif isinstance(rhs, Cost):
            time = self.time + rhs.time
            memory = self.memory + rhs.memory
            flops = self.flops + rhs.flops
            assert (time >= 0 and memory >= 0 and flops >= 0)
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, (OpCost, Cost))
        time = 0
        memory = 0
        flops = 0
        if isinstance(rhs, OpCost):
            time = self.cost.time - rhs.cost.time
            memory = self.cost.memory - rhs.cost.memory
            flops = self.cost.flops - rhs.cost.flops
            assert (time >= 0 and memory >= 0 and flops >= 0)
        elif isinstance(rhs, Cost):
            time = self.time - rhs.time
            memory = self.memory - rhs.memory
            flops = self.flops - rhs.flops
            assert (time >= 0 and memory >= 0 and flops >= 0)
        return Cost(time, memory, flops)


class CommOpCost(OpCost):
    OP_TYPE = "COMM"

    def __init__(self, op=None, op_desc=None, comm_context=None):
        super(CommOpCost, self).__init__(op=op, op_desc=op_desc)
        self._check_comm_op_type()
        self._comm_context = comm_context
        self._group_ranks = None
        self._comm_count = None
        self._hops = None
        self._rank_count = len(self.group_ranks)
        self._machine_count = None
        self._cost = self.calc_cost()

    @property
    def comm_context(self):
        return self._comm_context

    @property
    def comm_count(self):
        if self._comm_count is None:
            dtype = None
            shape = None
            if self.op is not None:
                vars = self.op.block.vars
                # NOTE: The tensor communicated input_name is "X" in default. Otherwise, this function should be overrided
                var_name = self.op.input("X")[0]
                var = vars[var_name]
                dtype = var.dtype
                shape = var.shape
            elif self.op_desc is not None:
                dtype = self.op_desc["inputs"]["X"][0][0]
                shape = self.op_desc["inputs"]["X"][0][1]

            factor = None
            if dtype == paddle.float32 or dtype == paddle.int32:
                factor = 4
            elif dtype == paddle.int64:
                factor = 8
            elif dtype == paddle.uint8:
                factor = 1
            elif dtype == paddle.float16:
                factor = 2
            else:
                raise TypeError(
                    "This dtype {} is not supported now".format(dtype))
            comm_count = reduce(lambda x, y: x * y, shape) * factor
            self._comm_count = comm_count

        return self._comm_count

    @property
    def rank_count(self):
        return self._rank_count

    @property
    def machine_count(self):
        if self._machine_count is None:
            cluster = self._comm_context.cluster
            self._machine_count = cluster.get_involved_machine_count(
                self.group_ranks)
        return self._machine_count

    @property
    def hops(self):
        if self._hops is None:
            self._hops = self.comm_context.get_hops(self.group_ranks)
        return self._hops

    @property
    def group_ranks(self):
        if self._group_ranks is None:
            if self.op_desc is not None:
                self._group_ranks = self.op_desc["group_ranks"]
            elif self.op is not None:
                ring_id = op.attrs("ring_id")
                process_group = get_process_group(ring_id)
                if process_group is None:
                    raise ValueError(
                        "There not exists process group whose ring_id is {}.".
                        format(ring_id))
                self._group_ranks = process_group.ranks
        return self._group_ranks

    @classmethod
    def _check_comm_op_type(cls):
        if cls.OP_TYPE != "COMM":
            if cls.OP_TYPE not in COMM_OP_TYPE:
                raise TypeError(
                    "Please Check op type in {}, but got {}.".format(
                        COMM_OP_TYPE, cls.OP_TYPE))


class CompOpCost(OpCost):
    OP_TYPE = "COMP"

    def __init__(self, op=None, op_desc=None, cluster=None):
        super(CompOpCost, self).__init__(op=op, op_desc=op_desc)
        self._check_comp_op_type()
        self._cost = self.calc_cost()
        self.cluster = cluster

    @classmethod
    def _check_comp_op_type(cls):
        if cls.OP_TYPE != "COMP":
            if cls.OP_TYPE in NON_COMP_TYPE:
                raise TypeError(
                    "Please Check op type not in {}, but got {}.".format(
                        NON_COMP_TYPE, cls.OP_TYPE))


def register_op_cost(cls):
    op_type = cls.OP_TYPE

    def register(op_type):
        global _g_op_cost_factory
        _g_op_cost_factory[op_type] = cls

    register(op_type)
    return cls


def calc_time_by_modeling(op=None, desc=None, cluster=None):
    op_type = op.type if op is not None else desc["op"]
    if op_type in COMM_OP_TYPE:
        op_cost = _g_op_cost_factory[op_type](op=op,
                                              op_desc=desc,
                                              comm_context=CommContext(cluster))
    elif op_type not in NON_COMP_TYPE:
        op_cost = _g_op_cost_factory[op_type](op=op,
                                              op_desc=desc,
                                              cluster=cluster)
    time = op_cost.calc_time()
    return time

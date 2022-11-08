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
# limitations under the License.

import warnings

import os
import paddle.fluid as fluid
from paddle.distributed import fleet
from paddle.fluid import core
from paddle.distributed.ps.utils.public import *  # noqa: F403
from paddle.fluid.framework import Program
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.executor import Executor
from paddle.fluid.parallel_executor import ParallelExecutor
from paddle.distributed.fleet.runtime.runtime_base import RuntimeBase
from paddle.distributed.fleet.base.private_helper_function import (
    wait_server_ready,
)
from paddle.distributed.fleet.proto import the_one_ps_pb2
from paddle.fluid.communicator import Communicator, HeterClient
from google.protobuf import text_format
from paddle.distributed.ps.coordinator import Coordinator

__all__ = [
    'Table',
    'SparseTable',
    'GeoSparseTable',
    'BarrierTable',
    'TensorTable',
    'DenseTable',
]


def get_program_by_id(context, program_id):
    programs = context["origin_main_programs"]
    for i, program in enumerate(programs):
        if id(program) == program_id:
            return program, context["origin_startup_programs"][i], i
    return None, None, None


def parse_table_class(varname, program_id, context):
    main_program, startup_program, idx = get_program_by_id(context, program_id)
    for op in main_program.global_block().ops:
        if not is_distributed_sparse_op(op) and not is_sparse_op(op):
            continue

        param_name = op.input("W")[0]

        if (
            param_name == varname
            and op.type == "lookup_table"
            or op.type == "lookup_table_v2"
        ):
            if op.has_attr('table_class') and op.attr("table_class") != "none":
                return op.attr('table_class')
            else:
                return "MemorySparseTable"


def check_embedding_dim(accessor_proto, varname, program_id, context):
    main_program, startup_program, idx = get_program_by_id(context, program_id)
    embedding_dim = 0
    for var in main_program.list_vars():
        if var.name == varname:
            embedding_dim = var.shape[1]
            print(
                'new var: {}, {}, {}'.format(
                    var, embedding_dim, accessor_proto.fea_dim
                )
            )
            break

    fea_dim = accessor_proto.fea_dim
    if accessor_proto.accessor_class == "SparseAccessor":
        if fea_dim != embedding_dim + 2:
            raise ValueError(
                "The fea_dim is wrong, it will be sparse_embedding_dim + 2: {}, but got {}".format(
                    embedding_dim + 2, fea_dim
                )
            )
    else:
        if fea_dim != embedding_dim:
            raise ValueError(
                "The fea_dim is wrong, it will be sparse_embedding_dim: {}, but got {}".format(
                    embedding_dim, fea_dim
                )
            )

    embedx_dim = accessor_proto.embedx_dim
    if accessor_proto.accessor_class == "SparseAccessor":
        if embedx_dim != embedding_dim - 1:
            raise ValueError(
                "The embedx_dim is wrong, it will be sparse_embedding_dim - 1: {}, but got {}".format(
                    embedding_dim - 1, embedx_dim
                )
            )
    else:
        if embedx_dim != embedding_dim - 3:
            raise ValueError(
                "The embedx_dim is wrong, it will be sparse_embedding_dim - 3: {}, but got {}".format(
                    embedding_dim - 3, embedx_dim
                )
            )


class Service:
    def __init__(self):
        pass

    def _set(self, service_proto):
        service_proto.server_class = "BrpcPsServer"
        service_proto.client_class = "BrpcPsClient"
        service_proto.service_class = "BrpcPsService"
        service_proto.start_server_port = 0
        service_proto.server_thread_num = 12


class GpuService(Service):
    def __init__(self):
        super().__init__()

    def _set(self, service_proto):
        service_proto.server_class = 'PsLocalServer'
        service_proto.client_class = 'PsLocalClient'


class Accessor:
    def __init__(self):
        self.accessor_class = ""
        self.optimizer = None
        self.feature_dim = 0
        self.embedding_dim = 0

    # TableAccessorParameter accessor
    def _set(
        self, accessor_proto, varname, program_id, context, common_accessor
    ):
        main_program, startup_program, idx = get_program_by_id(
            context, program_id
        )
        embedding_dim = 0
        for var in main_program.list_vars():
            if var.name == varname:
                embedding_dim = var.shape[1]
                break

        if not accessor_proto.HasField("accessor_class"):
            # DownpourSparseValueAccessor
            if context['use_ps_gpu']:
                accessor_proto.accessor_class = "CtrDymfAccessor"
            else:
                accessor_proto.accessor_class = "SparseAccessor"
        if not accessor_proto.HasField("fea_dim"):
            if accessor_proto.accessor_class == "SparseAccessor":
                accessor_proto.fea_dim = embedding_dim + 2
            else:
                accessor_proto.fea_dim = embedding_dim
        if not accessor_proto.HasField("embedx_dim"):
            if accessor_proto.accessor_class == "SparseAccessor":
                accessor_proto.embedx_dim = embedding_dim - 1
            else:
                accessor_proto.embedx_dim = embedding_dim - 3
        if not accessor_proto.HasField("embedx_threshold"):
            accessor_proto.embedx_threshold = 0

        graph_sgd_param = accessor_proto.graph_sgd_param
        if not graph_sgd_param.HasField("nodeid_slot"):
            graph_sgd_param.nodeid_slot = 9008
        if not graph_sgd_param.HasField("feature_learning_rate"):
            graph_sgd_param.feature_learning_rate = 0.05

        ctr_accessor_param = accessor_proto.ctr_accessor_param
        if accessor_proto.embedx_dim == 0:
            ctr_accessor_param.zero_init = False
        if not ctr_accessor_param.HasField("nonclk_coeff"):
            ctr_accessor_param.nonclk_coeff = 0.1
        if not ctr_accessor_param.HasField("click_coeff"):
            ctr_accessor_param.click_coeff = 1.0
        if not ctr_accessor_param.HasField("base_threshold"):
            ctr_accessor_param.base_threshold = 0
        if not ctr_accessor_param.HasField("delta_threshold"):
            ctr_accessor_param.delta_threshold = 0
        if not ctr_accessor_param.HasField("delta_keep_days"):
            ctr_accessor_param.delta_keep_days = 16
        if not ctr_accessor_param.HasField("show_click_decay_rate"):
            ctr_accessor_param.show_click_decay_rate = 1
        if not ctr_accessor_param.HasField("delete_threshold"):
            ctr_accessor_param.delete_threshold = 0
        if not ctr_accessor_param.HasField("delete_after_unseen_days"):
            ctr_accessor_param.delete_after_unseen_days = 30
        if not ctr_accessor_param.HasField("ssd_unseenday_threshold"):
            ctr_accessor_param.ssd_unseenday_threshold = 1

        for sgd_param in [
            accessor_proto.embed_sgd_param,
            accessor_proto.embedx_sgd_param,
        ]:
            if not sgd_param.HasField("name"):
                if common_accessor.accessor_class == "sgd":
                    sgd_param.name = "SparseNaiveSGDRule"
                if common_accessor.accessor_class == "adam":
                    sgd_param.name = "SparseAdamSGDRule"
                else:  # for fl-ps, because geo accessor is 'sum'
                    sgd_param.name = "SparseAdamSGDRule"

            if (
                sgd_param.name == "SparseAdaGradSGDRule"
                or sgd_param.name == "StdAdaGradSGDRule"
            ):
                if not sgd_param.adagrad.HasField("learning_rate"):
                    sgd_param.adagrad.learning_rate = 0.05
                if not sgd_param.adagrad.HasField("initial_g2sum"):
                    sgd_param.adagrad.initial_g2sum = 3.0
                if not sgd_param.adagrad.HasField("initial_range"):
                    sgd_param.adagrad.initial_range = 0.0001
                if len(sgd_param.adagrad.weight_bounds) == 0:
                    sgd_param.adagrad.weight_bounds.extend([-10.0, 10.0])

            if sgd_param.name == "SparseNaiveSGDRule":
                if not sgd_param.naive.HasField("learning_rate"):
                    learning_rate = common_accessor.initializers[-1].split("&")[
                        1
                    ]
                    sgd_param.naive.learning_rate = float(learning_rate)
                if not sgd_param.naive.HasField("initial_range"):
                    initial_range = common_accessor.initializers[0].split("&")[
                        -1
                    ]
                    sgd_param.naive.initial_range = float(initial_range)
                if len(sgd_param.naive.weight_bounds) == 0:
                    sgd_param.naive.weight_bounds.extend([-10.0, 10.0])

            if (
                sgd_param.name == "SparseAdamSGDRule"
                or sgd_param.name == "SparseSharedAdamSGDRule"
            ):
                if not sgd_param.adam.HasField("learning_rate"):
                    learning_rate = common_accessor.initializers[-1].split("&")[
                        1
                    ]
                    sgd_param.adam.learning_rate = float(learning_rate)
                if not sgd_param.adam.HasField("initial_range"):
                    initial_range = common_accessor.initializers[0].split("&")[
                        -1
                    ]
                    sgd_param.adam.initial_range = float(initial_range)

                attr_list = [x.split("&") for x in common_accessor.attrs]
                if (
                    not sgd_param.adam.HasField("beta1_decay_rate")
                    and common_accessor.accessor_class == "adam"
                ):
                    sgd_param.adam.beta1_decay_rate = float(attr_list[0][1])
                else:
                    sgd_param.adam.beta1_decay_rate = 0.9
                if (
                    not sgd_param.adam.HasField("beta2_decay_rate")
                    and common_accessor.accessor_class == "adam"
                ):
                    sgd_param.adam.beta2_decay_rate = float(attr_list[1][1])
                else:
                    sgd_param.adam.beta2_decay_rate = 0.999
                if (
                    not sgd_param.adam.HasField("ada_epsilon")
                    and common_accessor.accessor_class == "adam"
                ):
                    sgd_param.adam.ada_epsilon = float(attr_list[2][1])
                else:
                    sgd_param.adam.ada_epsilon = 1e-08
                if len(sgd_param.adam.weight_bounds) == 0:
                    sgd_param.adam.weight_bounds.extend([-10.0, 10.0])


class CommonAccessor(Accessor):
    def __init__(self):
        super().__init__()
        self.table_name = ''
        self.entry = 'none'
        self.attrs = []
        self.params = []
        self.dims = []
        self.trainer_num = 0
        self.sync = False
        self.initializers = []
        self.opt_input_map = {}
        self.opt_attr_map = {}
        self.opt_init_map = {}
        self.define_optimize_map()

    def define_optimize_map(self):
        opt_input_map = {}
        opt_input_map["sgd"] = [("Param", None), ("LearningRate", 1)]
        opt_input_map["adam"] = [
            ("Param", None),
            ("Moment1", None),
            ("Moment2", None),
            ("Beta1Pow", 1),
            ("Beta2Pow", 1),
            ("LearningRate", 1),
        ]
        opt_input_map["adam_d2sum"] = [
            ("Param", None),
            ("D2Sum", None),
            ("G2Sum", None),
            ("Moment", None),
            ("MomentDecayRate", 1),
            ("AdaDecayRate", 1),
            ("AdaEpsilon", 1),
            ("LearningRate", 1),
        ]
        opt_input_map["sum"] = [("Param", None)]
        opt_input_map["naive_adagrad"] = [
            ("Param", None),
            ("G2Sum", 1),
            ("LearningRate", 1),
        ]
        opt_input_map["summary"] = [("Param", None), ("SummaryDecayRate", 1)]

        opt_attr_map = {}
        opt_attr_map["sgd"] = []
        opt_attr_map["sum"] = []
        opt_attr_map["naive_adagrad"] = []
        opt_attr_map["adam"] = [
            ("beta1", "f"),
            ("beta2", "f"),
            ("epsilon", "f"),
        ]
        opt_attr_map["adam_d2sum"] = [
            ("beta1", "f"),
            ("beta2", "f"),
            ("epsilon", "f"),
        ]
        opt_attr_map["summary"] = [("summary_decay_rate", "f")]

        opt_init_map = {}
        opt_init_map["gaussian_random"] = ["seed", "mean", "std"]
        opt_init_map["fill_constant"] = ["value"]
        opt_init_map["uniform_random"] = ["seed", "min", "max"]
        opt_init_map["truncated_gaussian_random"] = ["seed", "mean", "std"]

        self.opt_attr_map = opt_attr_map
        self.opt_input_map = opt_input_map
        self.opt_init_map = opt_init_map

    def parse_entry(self, varname, program_id, context):
        main_program, startup_program, idx = get_program_by_id(
            context, program_id
        )
        for op in main_program.global_block().ops:
            if not is_distributed_sparse_op(op) and not is_sparse_op(op):
                continue

            param_name = op.input("W")[0]

            if param_name == varname and op.type == "lookup_table":
                self.entry = op.attr('entry')
                break

            if param_name == varname and op.type == "lookup_table_v2":
                self.entry = "none"
                break

    def get_shard(self, total_dim, shard_num, pserver_id):
        blocksize = int(total_dim / shard_num + 1)

        if blocksize * (pserver_id + 1) <= total_dim:
            return blocksize
        else:
            if blocksize * pserver_id < total_dim:
                return total_dim - blocksize * pserver_id
            else:
                return 0

    def get_initializer_attr(self, value_name, o_startup_program):
        l_in = "&"
        attr_str = ""

        origin_var_name = value_name
        # print("get_initializer_attr param name:", value_name)
        for op in o_startup_program.global_block().ops:
            if (
                op.type in self.opt_init_map.keys()
                and origin_var_name == op.output("Out")[0]
            ):
                init_attr = [op.type]
                # print("get_initializer_attr op type:", op.type)
                for attr in self.opt_init_map[op.type]:
                    # print("get_initializer_attr opt_init_map attr:", attr)
                    init_attr.append(str(op.attr(attr)))
                    # print("get_initializer_attr op attr:", str(op.attr(attr)))
                attr_str = l_in.join(init_attr)
                break
        return attr_str

    def parse_by_optimizer(self, ctx, context):
        grad_name = ctx.origin_varnames()[0]
        is_sparse = ctx.is_sparse()
        size = ctx.sections()[0]
        single_dim = ctx.sections()[1] if ctx.is_sparse() else 1
        adam_d2sum = context["user_defined_strategy"].adam_d2sum
        # print("parse_by_optimizer table_id:{} is_datanorm:{}".format(
        #     ctx.table_id(), ctx.is_datanorm_table()))

        main_program, startup_program, idx = get_program_by_id(
            context, ctx.program_id()
        )
        pserver_id = get_role_id(context['role_maker'])
        pserver_num = len(get_ps_endpoints(context['role_maker']))
        optimizer_ops = get_optimize_ops(main_program)
        # print("the one ps optimizer_ops:", optimizer_ops)
        # print("the one ps parse_by_optimizer grad_name:", grad_name)
        oop = None

        for op in optimizer_ops:
            if ("Param" in op.input_names) and (
                op.input("Param")[0]
                == context['grad_name_to_param_name'][grad_name]
            ):
                oop = op
                break

        if oop is None:
            raise ValueError("can not find optimizer for {}".format(grad_name))

        params = []
        dims = []
        attrs = []
        initializers = []

        self.trainer_num = get_trainers(context['role_maker'])
        self.table_num = size
        self.table_dim = single_dim

        if oop.type != 'adam' and adam_d2sum:
            print('optimization algorithm is not adam, set adam_d2sum False')
            adam_d2sum = False
        print("adam_d2sum:", adam_d2sum)
        if context['ps_mode'] == DistributedMode.GEO:
            param_varnames = self.opt_input_map["sum"]
            attr_varnames = self.opt_attr_map["sum"]
            self.accessor_class = "sum"
        elif context['use_ps_gpu'] and is_sparse:
            param_varnames = self.opt_input_map["naive_adagrad"]
            attr_varnames = self.opt_attr_map["naive_adagrad"]
            self.accessor_class = "sgd"
        elif ctx.is_datanorm_table():
            param_varnames = self.opt_input_map["summary"]
            attr_varnames = self.opt_attr_map["summary"]
            self.accessor_class = "summary"
        elif adam_d2sum and not is_sparse:
            param_varnames = self.opt_input_map["adam_d2sum"]
            attr_varnames = self.opt_attr_map["adam_d2sum"]
            self.accessor_class = "adam_d2sum"
        else:
            if oop.type != 'sgd' and oop.type != 'adam':
                raise ValueError(
                    "The dense optimizer in PS is only supported SGD or Adam!"
                )
            param_varnames = self.opt_input_map[oop.type]
            attr_varnames = self.opt_attr_map[oop.type]
            self.accessor_class = oop.type

        for (formal_name, shape) in param_varnames:
            params.append(formal_name)
            if self.accessor_class == "adam_d2sum":
                # for dims
                if shape is None:
                    if is_sparse:
                        shape = single_dim
                    else:
                        shape = self.get_shard(size, pserver_num, pserver_id)
                dims.append(shape)

                # for initializers
                if formal_name == "Param" or formal_name == "LearningRate":
                    param = main_program.global_block().vars[
                        oop.input(formal_name)[0]
                    ]
                    # TODO: for dense learning_rate, can be different from sparse lr
                    if (
                        formal_name == "LearningRate"
                        and param.name != "learning_rate_" + str(idx)
                    ):
                        warnings.warn("will support decay soon")
                        param = main_program.global_block().vars[
                            "learning_rate_" + str(idx)
                        ]

                    initializer = self.get_initializer_attr(
                        param.name, startup_program
                    )
                elif formal_name == "MomentDecayRate":
                    initializer = "fill_constant&0.99"
                elif formal_name == "AdaDecayRate":
                    initializer = "fill_constant&0.9999"
                elif formal_name == "AdaEpsilon":
                    initializer = "fill_constant&1.0e-8"
                else:
                    initializer = "fill_constant&0"
                initializers.append(initializer)
            elif self.accessor_class == "summary":
                # for dims
                if shape is None:
                    if is_sparse:
                        shape = single_dim
                    else:
                        shape = self.get_shard(size, pserver_num, pserver_id)
                dims.append(shape)

                # for initializers
                if formal_name == "Param":
                    param = main_program.global_block().vars[
                        oop.input(formal_name)[0]
                    ]

                    initializer = self.get_initializer_attr(
                        param.name, startup_program
                    )
                elif formal_name == "SummaryDecayRate":
                    initializer = "fill_constant&0.999999"
                else:
                    initializer = "fill_constant&0"
                initializers.append(initializer)
            else:
                if formal_name == "G2Sum":
                    dims.append(1)
                    initializer = "fill_constant&0"
                    initializers.append(initializer)
                else:
                    param = main_program.global_block().vars[
                        oop.input(formal_name)[0]
                    ]
                    if (
                        formal_name == "LearningRate"
                        and param.name != "learning_rate_" + str(idx)
                    ):
                        warnings.warn("will support decay soon")
                        param = main_program.global_block().vars[
                            "learning_rate_" + str(idx)
                        ]

                    if shape is None:
                        if is_sparse:
                            shape = single_dim
                        else:
                            shape = self.get_shard(
                                size, pserver_num, pserver_id
                            )
                    dims.append(shape)

                    initializer = self.get_initializer_attr(
                        param.name, startup_program
                    )
                    initializers.append(initializer)

        if self.accessor_class == 'summary':
            datanorm_ops = get_datanorm_ops(main_program)
            for op in datanorm_ops:
                if ("BatchSize" in op.input_names) and (
                    op.input("BatchSize")[0]
                    == context['grad_name_to_param_name'][grad_name]
                ):
                    oop = op
                    break

        for (attr_varname, type_) in attr_varnames:
            value = oop.attr(attr_varname)
            attrs.append("&".join([attr_varname, str(value)]))

        self.params = params
        self.dims = dims
        self.initializers = initializers
        self.attrs = attrs

    # CommonAccessorParameter common
    def _set(self, proto):
        proto.name = self.accessor_class
        proto.table_name = self.table_name
        proto.params.extend(self.params)
        proto.dims.extend(self.dims)
        proto.initializers.extend(self.initializers)
        proto.entry = self.entry
        proto.trainer_num = self.trainer_num
        proto.sync = self.sync
        proto.table_num = self.table_num
        proto.table_dim = self.table_dim
        proto.attr = "#".join(self.attrs)


class Tensor:
    def __init__(self, tesnor_dcit):
        self.tensor_dict = tesnor_dcit

    def _set(self, tensor_proto):
        tensor_proto.main_program_id = self.tensor_dict.get(
            "main_program_id", 0
        )
        tensor_proto.startup_program_id = self.tensor_dict.get(
            "startup_program_id", 0
        )
        tensor_proto.feed_var_name = self.tensor_dict.get("feed_var_name", '')
        tensor_proto.fetch_var_name = self.tensor_dict.get("fetch_var_name", '')
        tensor_proto.tensor_table_class = self.tensor_dict.get(
            "tensor_table_class", ''
        )


class Table:
    def __init__(self):
        self.table_class = None
        self.shard_num = -1
        self.type = None
        self.accessor = Accessor()
        self.shard_num = 256
        self.common = CommonAccessor()
        self.tensor = None

    def _set(self, table_proto):
        pass


class BarrierTable(Table):
    def __init__(self, context, idx):
        super().__init__()
        self.type = None
        self.shard_num = 256
        self.accessor.accessor_class = 'CommMergeAccessor'
        self.common.attrs = ""
        self.common.dims = []
        self.common.params = []
        self.is_heter_ps_mode = context['is_heter_ps_mode']
        self.role_maker = context['role_maker']
        self.idx = idx
        self.is_sync = context['is_sync']

    def _set(self, table_proto):
        table_proto.table_id = self.idx
        table_proto.table_class = 'BarrierTable'
        table_proto.shard_num = 256
        table_proto.type = the_one_ps_pb2.PS_OTHER_TABLE

        table_proto.accessor.accessor_class = "CommMergeAccessor"
        table_proto.accessor.fea_dim = 0
        table_proto.accessor.embedx_dim = 0

        table_proto.common.name = ""
        table_proto.common.table_name = "barrier_table"
        table_proto.common.sync = self.is_sync
        table_proto.common.entry = 'none'

        trainer_num = get_trainers(self.role_maker)
        if self.is_heter_ps_mode:
            trainer_num += len(self.role_maker._get_heter_worker_endpoints())
        table_proto.common.trainer_num = trainer_num


class TensorTable(Table):
    def __init__(self, idx, tensor_dict, role_maker):
        super().__init__()
        self.idx = idx
        self.tensor_dict = tensor_dict
        self.role_maker = role_maker

    def _set(self, table_proto):
        table_proto.table_id = self.idx
        table_proto.type = the_one_ps_pb2.PS_OTHER_TABLE
        table_proto.table_class = self.tensor_dict.get("tensor_table_class", '')

        table_proto.accessor.accessor_class = "CommMergeAccessor"

        table_proto.common.table_name = self.tensor_dict.get(
            "feed_var_name", ''
        )
        table_proto.common.trainer_num = get_trainers(self.role_maker)

        tensor = Tensor(self.tensor_dict)
        tensor._set(table_proto.tensor)


class SparseTable(Table):
    def __init__(self, context, send_ctx):
        super().__init__()
        self.context = context
        self.ctx = send_ctx
        self.type = None
        self.table_class = 'MemorySparseTable'
        self.accessor = Accessor()

    def _set(self, table_proto):
        ctx = self.ctx
        if (
            ctx.is_tensor_table()
            or len(ctx.origin_varnames()) < 1
            or (not ctx.is_sparse())
        ):
            return
        table_proto.table_id = ctx.table_id()
        table_proto.table_class = self.table_class
        table_proto.type = the_one_ps_pb2.PS_SPARSE_TABLE
        table_proto.shard_num = self.shard_num
        if table_proto.sparse_table_cache_file_num > len(
            get_ps_endpoints(self.context['role_maker'])
        ):
            table_proto.sparse_table_cache_file_num = len(
                get_ps_endpoints(self.context['role_maker'])
            )

        self.common.table_name = self.context['grad_name_to_param_name'][
            ctx.origin_varnames()[0]
        ]

        self.common.parse_by_optimizer(ctx, self.context)
        self.common.parse_entry(
            self.common.table_name, ctx.program_id(), self.context
        )
        self.common.sync = True if self.context['is_sync'] else False

        self.common._set(table_proto.common)

        print('new table_name: {}'.format(self.common.table_name))
        all_table_proto = self.context[
            "user_defined_strategy"
        ].sparse_table_configs
        usr_table_proto = all_table_proto.add()
        for proto in all_table_proto:
            if proto.table_name == self.common.table_name:
                usr_table_proto = proto
                break
        if usr_table_proto.HasField("table_class"):
            table_proto.table_class = usr_table_proto.table_class
        else:
            table_proto.table_class = 'MemorySparseTable'
            warnings.warn("The PS mode must use MemorySparseTable.")
        if usr_table_proto.HasField("shard_num"):
            table_proto.shard_num = usr_table_proto.shard_num
        else:
            if self.context['use_ps_gpu']:
                table_proto.shard_num = 37
                warnings.warn(
                    "The shard_num of sparse table is not set, use default value 37 in gpups."
                )
            else:
                table_proto.shard_num = 1000
                warnings.warn(
                    "The shard_num of sparse table is not set, use default value 1000 in cpups."
                )

        if usr_table_proto.HasField("enable_sparse_table_cache"):
            table_proto.enable_sparse_table_cache = (
                usr_table_proto.enable_sparse_table_cache
            )
        if usr_table_proto.HasField("sparse_table_cache_rate"):
            table_proto.sparse_table_cache_rate = (
                usr_table_proto.sparse_table_cache_rate
            )
        if usr_table_proto.HasField("sparse_table_cache_file_num"):
            table_proto.sparse_table_cache_file_num = (
                usr_table_proto.sparse_table_cache_file_num
            )
        if usr_table_proto.HasField("enable_revert"):
            table_proto.enable_revert = usr_table_proto.enable_revert
        if usr_table_proto.HasField("shard_merge_rate"):
            table_proto.shard_merge_rate = usr_table_proto.shard_merge_rate

        if usr_table_proto.accessor.ByteSize() == 0:
            warnings.warn(
                "The accessor of sparse table is not set, use default value."
            )

        table_proto.accessor.ParseFromString(
            usr_table_proto.accessor.SerializeToString()
        )
        self.accessor._set(
            table_proto.accessor,
            self.common.table_name,
            ctx.program_id(),
            self.context,
            self.common,
        )

        check_embedding_dim(
            table_proto.accessor,
            self.common.table_name,
            ctx.program_id(),
            self.context,
        )


class GeoSparseTable(SparseTable):
    def __init__(self, context, send_ctx):
        super().__init__(context, send_ctx)
        self.table_class = "MemorySparseGeoTable"
        if self.context['ps_mode'] != DistributedMode.GEO:
            raise ValueError("not geo sparse table!")

    def _set(self, table_proto):
        ctx = self.ctx
        if (
            ctx.is_tensor_table()
            or len(ctx.origin_varnames()) < 1
            or (not ctx.is_sparse())
        ):
            return
        table_proto.table_id = ctx.table_id()
        table_proto.table_class = self.table_class
        table_proto.type = the_one_ps_pb2.PS_SPARSE_TABLE
        table_proto.shard_num = self.shard_num

        table_proto.accessor.accessor_class = 'CommMergeAccessor'
        table_proto.accessor.fea_dim = ctx.sections()[0]
        table_proto.accessor.embedx_dim = ctx.sections()[1]

        self.common.table_name = self.context['grad_name_to_param_name'][
            ctx.origin_varnames()[0]
        ]
        self.common.parse_by_optimizer(ctx, self.context)
        self.common.parse_entry(
            self.common.table_name, ctx.program_id(), self.context
        )
        self.common.sync = False
        self.common._set(table_proto.common)


class DenseTable(Table):
    def __init__(self, context, send_ctx):
        super().__init__()
        self.context = context
        self.ctx = send_ctx
        self.accessor = Accessor()

    def _set(self, table_proto):
        ctx = self.ctx
        if (
            ctx.is_tensor_table()
            or len(ctx.origin_varnames()) < 1
            or (ctx.is_sparse())
        ):
            return

        table_proto.table_id = ctx.table_id()

        table_proto.type = the_one_ps_pb2.PS_DENSE_TABLE
        table_proto.table_class = "MemoryDenseTable"
        table_proto.shard_num = 256

        table_proto.accessor.accessor_class = 'CommMergeAccessor'
        table_proto.accessor.fea_dim = ctx.sections()[0]
        table_proto.accessor.embedx_dim = 1

        self.common.table_name = "MergedDense"
        self.common.parse_by_optimizer(ctx, self.context)
        self.common.parse_entry(
            self.common.table_name, ctx.program_id(), self.context
        )
        self.common.sync = True if self.context['is_sync'] else False

        self.common._set(table_proto.common)


class Server:
    def __init__(self):
        pass

    def _set(self):
        pass


class DownpourServer(Server):
    def __init__(self):
        super().__init__()

    def _set(self):
        pass


class Worker:
    def __init__(self):
        pass

    def _set(self):
        pass


class DownpourWorker(Worker):
    def __init__(self):
        super().__init__()

    def _set(self):
        pass


class fsClient:
    def __init__(self, fs_client_param):
        self.fs_client_param = fs_client_param

    def _set(self, proto):
        if not text_format.MessageToString(self.fs_client_param):
            return
        proto.uri = self.fs_client_param.uri
        proto.user = self.fs_client_param.user
        proto.passwd = self.fs_client_param.passwd
        proto.hadoop_bin = self.fs_client_param.hadoop_bin


class PsDescBuilder:
    def __init__(self, context):
        self.context = context
        self.is_sync = context['is_sync']
        self.ps_mode = context['ps_mode']
        self.is_heter_ps_mode = context['is_heter_ps_mode']
        self.use_ps_gpu = context['use_ps_gpu']
        self.barrier_table_id = None

        self.send_ctx = get_the_one_send_context(
            self.context, split_dense_table=self.is_heter_ps_mode
        )

        self.tensor_table_dict = {}  # TODO
        self._server_sub_program = []

        self.tables = self._get_tables()

        self.service = self._get_service()
        self.fs_client = self._get_fs_client()

        self.ps_desc = the_one_ps_pb2.PSParameter()
        self.fl_desc = the_one_ps_pb2.FLParameter()

    def _get_tensor_tables(self):
        program_idx = 0
        if not self.tensor_table_dict:
            self._server_sub_program.append(Program().desc)
        tables = []
        for table_name in self.tensor_table_dict:
            tables.append(
                globals()['TensorTable'](
                    len(tables), tensor_dict, self.context['role_maker']
                )
            )
            program_idx += 1
        return tables

    def _get_tables(self):
        tables = []
        for idx, (name, ctx) in enumerate(self.send_ctx.items()):
            print("idx, name, ctx:", idx, name, ctx)
            if ctx.is_sparse():
                if self.ps_mode == DistributedMode.GEO:
                    if (
                        self.context['local_sparse']
                        and name[:-5] in self.context['local_sparse']
                    ) or (not self.context['local_sparse']):
                        tables.append(
                            globals()['GeoSparseTable'](self.context, ctx)
                        )
                    else:
                        tables.append(
                            globals()['SparseTable'](self.context, ctx)
                        )
                else:
                    tables.append(globals()['SparseTable'](self.context, ctx))
            else:
                tables.append(globals()['DenseTable'](self.context, ctx))
        self.tensor_tables = self._get_tensor_tables()
        tables.extend(self.tensor_tables)
        tables.append(globals()['BarrierTable'](self.context, len(tables)))
        return tables

    def _get_service(self):
        if self.use_ps_gpu:
            return GpuService()
        else:
            return Service()

    def _get_fs_client(self):
        return fsClient(self.context["user_defined_strategy"].fs_client_param)

    def build_fl_client_desc(self, client_info):
        pass

    def build_worker_desc(self):
        for table in self.tables:
            table_proto = (
                self.ps_desc.worker_param.downpour_worker_param.downpour_table_param.add()
            )
            table._set(table_proto)
            table_proto = (
                self.ps_desc.server_param.downpour_server_param.downpour_table_param.add()
            )
            table._set(table_proto)
            if type(table) == BarrierTable and self.barrier_table_id is None:
                self.barrier_table_id = table.idx
        self.service._set(
            self.ps_desc.server_param.downpour_server_param.service_param
        )
        self.fs_client._set(self.ps_desc.fs_client_param)
        return text_format.MessageToString(self.ps_desc)

    def build_server_desc(self):
        self.sparse_table_maps = {}
        for table in self.tables:
            table_proto = (
                self.ps_desc.server_param.downpour_server_param.downpour_table_param.add()
            )
            table._set(table_proto)
            if (
                table_proto.type == the_one_ps_pb2.PS_SPARSE_TABLE
                and table_proto.common is not None
            ):
                self.sparse_table_maps[
                    table_proto.common.table_name
                ] = table_proto.table_id

        self.service._set(
            self.ps_desc.server_param.downpour_server_param.service_param
        )
        self.fs_client._set(self.ps_desc.fs_client_param)
        return text_format.MessageToString(self.ps_desc)


class TheOnePSRuntime(RuntimeBase):
    def __init__(self):
        super().__init__()
        self._communicator = None
        self._server = None
        self._worker = fluid.core.DistFleetWrapper()
        self._coordinator = None
        self._server_sub_program = []
        self._heter_client = None
        self._send_ctx = None

    def _set_basic_info(self, context):
        self.context = context
        self.role_maker = context["role_maker"]
        self.role_id = get_role_id(self.role_maker)
        self.debug = bool(int(os.getenv("PSERVER_DEBUG", "0")))

        self.origin_main_program = context["origin_main_program"]
        self.origin_main_programs = context.get(
            "origin_main_programs", [self.origin_main_program]
        )
        self.context["origin_main_programs"] = self.origin_main_programs
        self.context["origin_startup_programs"] = context.get(
            'origin_startup_programs', [context['origin_startup_program']]
        )
        self.context[
            'is_heter_ps_mode'
        ] = self.role_maker._is_heter_parameter_server_mode
        self.is_heter_ps_mode = self.context['is_heter_ps_mode']
        self.context['trainer'] = TrainerRuntimeConfig(
            context['valid_strategy']
        )
        self.context['ps_mode'] = self.context['trainer'].mode
        self.context['use_ps_gpu'] = context['valid_strategy'].a_sync_configs[
            'use_ps_gpu'
        ]
        self.context['is_sync'] = (
            True if self.context['ps_mode'] == DistributedMode.SYNC else False
        )
        self.context['grad_name_to_param_name'] = {}
        self.context['tensor_table'] = {}
        # FL
        self.context['local_sparse'] = context[
            "user_defined_strategy"
        ].trainer_desc_configs["local_sparse"]
        self.context['remote_sparse'] = context[
            "user_defined_strategy"
        ].trainer_desc_configs["remote_sparse"]
        print(
            "fl-ps > local_sparse: {}, remote_sparse: {}".format(
                self.context['local_sparse'], self.context['remote_sparse']
            )
        )

        build_var_distributed(self.context)

        self.trainer_endpoints = get_trainer_endpoints(self.role_maker)

        self.endpoints = get_ps_endpoints(self.role_maker)
        self.string_hosts = []
        for idx, ep in enumerate(self.endpoints):
            host, port = ep.split(":")
            pshost = fluid.core.PSHost(host, int(port), idx)
            self.string_hosts.append(pshost.serialize_to_string())

        self.with_coordinator = self.role_maker._with_coordinator
        self.coordinator_hosts = []
        if self.with_coordinator:
            print("fl-ps > all ps addrs: {}".format(self.string_hosts))
            coordinator_endpoints = self.role_maker._get_coordinator_endpoints()
            for idx, ep in enumerate(coordinator_endpoints):
                ip, port = ep.split(":")
                pshost = fluid.core.PSHost(ip, int(port), idx)
                self.coordinator_hosts.append(pshost.serialize_to_string())

        self.ps_desc_builder = PsDescBuilder(self.context)

    def _init_all_params(self, scopes, send_ctx, recv_map):
        all_var_names = []
        for name, ctx in send_ctx.items():
            if ctx.is_sparse():
                continue
            _, _, idx = get_program_by_id(self.context, ctx.program_id())
            scope = scopes[idx]
            table_id = ctx.table_id()
            var_names = recv_map[table_id]
            # print("init params:", idx, table_id, var_names)
            self._worker.push_dense_params(scope, table_id, var_names)
            all_var_names.extend(var_names)
        return all_var_names

    def _pull_all_dense(self, scopes, send_ctx, recv_map):
        all_var_names = []
        for name, ctx in send_ctx.items():
            if ctx.is_sparse():
                continue
            _, _, idx = get_program_by_id(self.context, ctx.program_id())
            scope = scopes[idx]
            table_id = ctx.table_id()
            var_names = recv_map[table_id]
            # print("pull all dense:", idx, table_id, var_names)
            self._worker.pull_dense_params(scope, table_id, var_names)
            all_var_names.extend(var_names)
        return all_var_names

    def _init_params(self, program, scope, send_ctx, recv_map):
        all_var_names = []
        for name, ctx in send_ctx.items():
            if ctx.is_sparse():
                continue
            if ctx.program_id() != id(program):
                continue
            table_id = ctx.table_id()
            var_names = recv_map[table_id]
            # print("init params:", table_id, var_names)
            self._worker.push_dense_params(scope, table_id, var_names)
            all_var_names.extend(var_names)
        return all_var_names

    def _pull_dense(self, program, scope, send_ctx, recv_map):
        all_var_names = []
        for name, ctx in send_ctx.items():
            if ctx.is_sparse():
                continue
            if ctx.program_id() != id(program):
                continue
            table_id = ctx.table_id()
            var_names = recv_map[table_id]
            # print("pull dense:", table_id, var_names)
            self._worker.pull_dense_params(scope, table_id, var_names)
            all_var_names.extend(var_names)
        return all_var_names

    def _init_worker(self, scopes=None):
        worker_desc = self.ps_desc_builder.build_worker_desc()
        if self.context['use_ps_gpu']:
            main_program = self.context['loss'].block.program
            if not main_program._fleet_opt:
                main_program._fleet_opt = {}
            main_program._fleet_opt["use_ps_gpu"] = True
            gpus_env = os.getenv("FLAGS_selected_gpus")
            gpus_env = [int(s) for s in gpus_env.split(",")]
            main_program._fleet_opt["worker_places"] = gpus_env
            PSGPU = fluid.core.PSGPU()
            PSGPU.init_gpu_ps(gpus_env)

        def sync_strategy_envs():
            kwargs = {}
            kwargs[
                "pserver_endpoints"
            ] = self.role_maker._get_pserver_endpoints()
            kwargs["trainer_id"] = self.role_maker._worker_index()
            return kwargs

        dense_map = get_the_one_recv_context(
            self.context, split_dense_table=self.is_heter_ps_mode
        )
        send_ctx = get_the_one_send_context(
            self.context,
            split_dense_table=self.is_heter_ps_mode,
            ep_list=self.endpoints,
        )
        self._send_ctx = send_ctx
        trainer_config = self.context['trainer']

        if self.debug:
            print("worker_desc: \n{}".format(worker_desc))
            print("communicator send_ctx:")
            for key in send_ctx:
                print("{}: {}".format(key, send_ctx[key]))
            for key in dense_map:
                print("{}: {}".format(key, dense_map[key]))

        kwargs = {}
        kwargs['need_global_step'] = "0"
        kwargs["trainer_id"] = self.role_maker._role_id()
        kwargs["trainers"] = self.role_maker._worker_num()

        kwargs["barrier_table_id"] = self.ps_desc_builder.barrier_table_id

        if self.context['ps_mode'] == DistributedMode.SYNC:
            sync_kwargs = sync_strategy_envs()
            kwargs.update(sync_kwargs)

        print("communicator config:", trainer_config.get_communicator_flags())

        self._worker.init_worker(worker_desc, self.string_hosts, self.role_id)
        if not self.is_heter_ps_mode:
            self.trainer_endpoint = get_trainer_endpoint(self.role_maker)
            print("fl-ps > trainer_endpoint: {}".format(self.trainer_endpoint))
        print("fl-ps > with_coordinator? {}".format(self.with_coordinator))
        print("fl-ps > coordinator addr: {}".format(self.coordinator_hosts))
        if self.with_coordinator:
            self._worker.init_fl_worker(
                self.coordinator_hosts, self.role_id, self.trainer_endpoint
            )

        if (
            self.context['ps_mode'] == DistributedMode.GEO
            or self.is_heter_ps_mode
        ):
            self._communicator = Communicator(
                trainer_config.mode,
                kwargs,
                trainer_config.get_communicator_flags(),
            )
            self._communicator.init_with_ctx(
                send_ctx,
                dense_map,
                worker_desc,
                self.string_hosts,
                fluid.global_scope(),
            )
        fleet.util.barrier()

        # info = self._communicator.get_client_info()
        info = self._worker.get_client_info()
        if isinstance(info, list) and len(info) > 0:
            all_info = self.role_maker._all_gather(
                info[0]
            )  # 收集其他 client 的 service 地址
            # for unittest
            if not isinstance(all_info, list):
                warnings.warn("gloo may not initialize correctly")
                all_info = [all_info]

            # self._communicator.set_clients(all_info)
            # self._communicator.create_client_to_client_connection()
            self._worker.set_clients(all_info)
            self._worker.create_client2client_connection()
            print('create c2c connection done')
        else:
            print('cannot create c2c connection')

        dist_strategy = self.context["valid_strategy"]

        is_test = bool(int(os.getenv("TEST_MODE", "0")))

        if scopes is None:
            if len(self.origin_main_programs) > 1:
                raise ValueError(
                    "You must set the scope list when you have Multiple programs"
                )
            scopes = [fluid.global_scope()]
        if len(self.origin_main_programs) != len(scopes):
            raise VauleError("len(programs) != len(scopes)")

        self.scopes = scopes
        if not is_test:
            if (
                self.context['ps_mode'] == DistributedMode.GEO
                or self.is_heter_ps_mode
            ):
                self._communicator.init_params(dense_map)
            else:
                if not self.context['use_ps_gpu']:
                    if self.role_id == 0:
                        print("entering self._init_all_params()")
                        self._init_all_params(scopes, send_ctx, dense_map)

            fleet.util.barrier()  # 保证 0 号 worker 参数 push_dense_param over

        if not self.context['use_ps_gpu']:
            self._pull_all_dense(scopes, send_ctx, dense_map)
        fleet.util.barrier()

        if (
            self.context['ps_mode'] == DistributedMode.GEO
            or self.is_heter_ps_mode
        ):
            if not self._communicator.is_running():
                self._communicator.start()
            else:
                warnings.warn("communicator has been initialized, skip")

        launch_barrier = dist_strategy.a_sync_configs["launch_barrier"]
        launch_barrier_flag = int(os.getenv("FLAGS_LAUNCH_BARRIER", "1"))
        if launch_barrier and launch_barrier_flag:
            wait_server_ready(self.role_maker._get_pserver_endpoints())
            if (
                self.is_heter_ps_mode
                and self.role_maker._get_next_trainers() != []
            ):
                wait_server_ready(self.role_maker._get_next_trainers())
            if self.is_heter_ps_mode:
                previous_trainers = []
                if self.role_maker._get_previous_trainers() != []:
                    previous_trainers = self.role_maker._get_previous_trainers()
                next_trainers = []
                if self.role_maker._get_next_trainers() != []:
                    next_trainers = self.role_maker._get_next_trainers()
                self._heter_client = HeterClient(
                    next_trainers, previous_trainers, self.role_maker._role_id()
                )  # --> HeterClient::GetInstance

    def _init_coordinator(self, scopes=None):
        if self._coordinator is None:
            self._coordinator = Coordinator(self.string_hosts)

        print(">>> curr node ip: {}".format(self.coordinator_hosts[0]))
        print(">>> all trainer endpoints: {}".format(self.trainer_endpoints))
        self._coordinator.start_coordinator(
            self.coordinator_hosts[0], self.trainer_endpoints
        )

    def _make_fl_strategy(self):
        if self._coordinator is None:
            assert "Coordinator py object is null!"
        else:
            self._coordinator.make_fl_strategy()

    def _init_server(self, dirname=None, var_names=None, **kwargs):
        server_desc = self.ps_desc_builder.build_server_desc()
        trainers = get_trainers(self.role_maker)
        if self.is_heter_ps_mode:
            trainers += len(self.role_maker._get_heter_worker_endpoints())

        if self.debug:
            print("server_desc: \n{}".format(server_desc))

        self._server = fluid.core.DistFleetWrapper()
        self._server.init_server(
            server_desc,
            self.string_hosts,
            self.role_id,
            trainers,
            self._server_sub_program,
        )

        dist_varnames = get_sparse_tablenames(self.origin_main_programs, True)
        sparse_varnames = get_sparse_tablenames(
            self.origin_main_programs, False
        )

        distributed_varnames = dist_varnames + sparse_varnames

        if var_names is None:
            load_varnames = distributed_varnames
        else:
            for var_name in var_names:
                if var_name not in distributed_varnames:
                    raise ValueError(
                        "fleet.init server can only load sparse variables in {}".format(
                            distributed_varnames
                        )
                    )
            load_varnames = var_names

        if dirname is None or not load_varnames:
            return

        sparse_table_maps = self.ps_desc_builder.sparse_table_maps

        dirname = os.path.normpath(dirname)
        pserver_id = self.role_maker._role_id()

        for var_name in load_varnames:
            table_id = sparse_table_maps[var_name]
            self._server.load_sparse(dirname, "0", table_id)

    def _run_server(self):
        ep = get_ps_endpoint(self.role_maker)
        host, port = ep.split(":")
        self._server.run_server(host, int(port))

    def _stop_worker(self):
        if self.context['ps_mode'] == DistributedMode.GEO:
            self._communicator.stop()
        self._worker.stop_worker()
        if self.is_heter_ps_mode:
            assert (
                self._heter_client is not None
            ), "heter client should not be None in heterps mode"
            self._heter_client.stop()

    @staticmethod
    def __exclude_vars(exclude_var_names=[]):
        def is_valid(var):
            if var.name in exclude_var_names:
                return False

            from .utils.public import _get_varname_parts

            origin_varname, _, _ = _get_varname_parts(var.name)
            if origin_varname.endswith("@GRAD"):
                return False

            if origin_varname.startswith("learning_rate_"):
                return False

            if (
                var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH
                or var.desc.type() == core.VarDesc.VarType.FETCH_LIST
                or var.desc.type() == core.VarDesc.VarType.READER
            ):
                return False
            return var.persistable

        return is_valid

    def _get_inference_model_path(self, dirname):
        if dirname.startswith("afs:") or dirname.startswith("hdfs:"):
            model_path = "./dnn_plugin"
        else:
            model_path = os.path.join(dirname, "dnn_plugin")
        return model_path

    def _ps_save_dense_params(
        self, executor, dirname, scope, program, var_names=None
    ):
        dense_map = get_the_one_recv_context(
            self.context, split_dense_table=self.is_heter_ps_mode
        )
        send_ctx = get_the_one_send_context(
            self.context,
            split_dense_table=self.is_heter_ps_mode,
            ep_list=self.endpoints,
        )
        if program is None or len(self.origin_main_programs) == 1:
            program = self.origin_main_programs[0]
        dense_var_names = self._pull_dense(program, scope, send_ctx, dense_map)
        save_var_names = dense_var_names if var_names is None else var_names
        vars = [program.global_block().var(i) for i in save_var_names]
        import paddle

        with paddle.static.scope_guard(scope):
            paddle.static.save_vars(
                executor, "./", program, vars=vars, filename=dirname
            )

    def _save_sparse_params(
        self, executor, dirname, context, main_program, mode
    ):
        distributed_varnames = get_sparse_tablenames(
            self.origin_main_programs, True
        )
        values = []
        model_path = self._get_inference_model_path(dirname)
        for id, names in context.items():
            if names[0] not in distributed_varnames:
                # only save sparse param to local
                try:
                    self._worker.recv_and_save_model(id, model_path)
                except:
                    pass
            # save sparse & distributed param on server
            self._worker.save_one_model(id, dirname, mode)
            values.extend(names)
        # self._worker.save_all_model(dirname, mode)
        return values

    def _save_distributed_persistables(
        self, executor, dirname, main_program=None, mode=0, **kwargs
    ):
        """
        This function filters out all variables with `persistable==True` from the
        give `main_program` and then saves these variables to the folder `dirname`
        or file `filename`.

        The `dirname` is used to specify the folder where persistable variables
        are going to be saved. If you would like to save variables in separate
        files, set `filename` None; if you would like to save all variables in a
        single file, use `filename` to specify the file name.
        """

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type"
            )

        if main_program is None:
            main_program = self.context['origin_main_program']

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        self._worker.save_all_model(dirname, mode)

    def _ps_inference_save_inference_model(
        self,
        executor,
        dirname,
        feeded_var_names,
        target_vars,
        main_program=None,
        export_for_deployment=True,
        mode=0,
    ):
        """
        Prune the given `main_program` to build a new program especially for inference,
        and then save it and all related parameters to given `dirname` by the `executor`.
        """

        if isinstance(executor, ParallelExecutor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type, ParallelExecutor is not allowed"
            )

        if not isinstance(executor, Executor):
            raise TypeError(
                "in fleet.save() function, executor must be as Executor type"
            )

        import paddle

        program = (
            self.origin_main_programs[0]
            if main_program is None
            else main_program
        )
        _, _, idx = get_program_by_id(self.context, id(program))
        scope = self.scopes[idx]
        print("save inference model scope idx:", idx)

        if isinstance(program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        feed_vars = [
            program.global_block().var(name) for name in feeded_var_names
        ]

        infer_program = paddle.static.normalize_program(
            program, feed_vars, target_vars
        )

        infer_program._copy_dist_param_info_from(program)

        model_path = self._get_inference_model_path(dirname)
        model_basename = "__model__"
        model_basename = os.path.join(model_path, model_basename)
        paddle.save(infer_program, model_basename)

        sparses = get_the_one_recv_context(
            self.context,
            is_dense=False,
            split_dense_table=self.is_heter_ps_mode,
        )
        sparse_names = self._save_sparse_params(
            executor, dirname, sparses, main_program, mode
        )

        dense_map = get_the_one_recv_context(
            self.context, split_dense_table=self.is_heter_ps_mode
        )
        send_ctx = get_the_one_send_context(
            self.context,
            split_dense_table=self.is_heter_ps_mode,
            ep_list=self.endpoints,
        )
        self._pull_dense(program, scope, send_ctx, dense_map)

        generate_vars = self.context[
            "user_defined_strategy"
        ].trainer_desc_configs["stat_var_names"]
        generate_vars = [var for var in generate_vars]
        remaining_vars = list(
            filter(
                TheOnePSRuntime.__exclude_vars(sparse_names),
                infer_program.list_vars(),
            )
        )

        for var in remaining_vars:
            tensor = var.get_value(scope)
            paddle.save(
                tensor,
                os.path.join(model_path, var.name),
                use_binary_format=True,
            )

    def _save_cache_model(self, dirname, **kwargs):
        mode = kwargs.get("mode", 1)
        table_id = kwargs.get("table_id", 0)
        self._worker.client_flush()
        fleet.util.barrier()
        cache_threshold = 0.0

        if self.role_maker._is_first_worker():
            cache_threshold = self._worker.get_cache_threshold(table_id)
        # check cache threshold right or not
        fleet.util.barrier()

        if self.role_maker._is_first_worker():
            self._worker.cache_shuffle(table_id, dirname, mode, cache_threshold)

        fleet.util.barrier()

        feasign_num = -1
        if self.role_maker._is_first_worker():
            feasign_num = self._worker.save_cache(table_id, dirname, mode)

        fleet.util.barrier()
        return feasign_num

    def _check_save_pre_patch_done(self):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._worker.check_save_pre_patch_done()
        fleet.util.barrier()

    def _load_sparse_params(self, dirname, context, main_program, mode):
        distributed_varnames = get_sparse_tablenames(
            self.origin_main_programs, True
        )
        values = []
        for id, names in context.items():
            if names[0] not in distributed_varnames:
                # TODO: only load sparse param from local
                warnings.warn("varname is not in distributed_varnames, pass")
            # load sparse & distributed param on server
            self._worker.load_one_table(id, dirname, mode)
            values.extend(names)
        return values

    def _ps_inference_load_inference_model(
        self, dirname, mode=0, main_program=None
    ):
        main_program = (
            self.origin_main_programs[0]
            if main_program is None
            else main_program
        )
        _, _, idx = get_program_by_id(self.context, id(main_program))
        scope = self.scopes[idx]
        print("load inference model scope idx:", idx)

        if isinstance(main_program, CompiledProgram):
            raise TypeError(
                "in fleet.save() function, main_program must be as Program type, CompiledProgram is not allowed"
            )

        sparses = get_the_one_recv_context(
            self.context,
            is_dense=False,
            split_dense_table=self.is_heter_ps_mode,
        )

        sparse_varnames = self._load_sparse_params(
            dirname, sparses, main_program, mode
        )

        dense_map = get_the_one_recv_context(
            self.context, split_dense_table=self.is_heter_ps_mode
        )
        send_ctx = get_the_one_send_context(
            self.context,
            split_dense_table=self.is_heter_ps_mode,
            ep_list=self.endpoints,
        )

        recv_dense_varnames = []
        for _, names in dense_map.items():
            recv_dense_varnames.extend(names)

        loaded_varnames = sparse_varnames

        remaining_vars = list(
            filter(
                TheOnePSRuntime.__exclude_vars(loaded_varnames),
                main_program.list_vars(),
            )
        )

        model_path = self._get_inference_model_path(dirname)
        import paddle

        for var in remaining_vars:
            if var.name not in recv_dense_varnames:
                continue
            tensor = paddle.load(os.path.join(model_path, var.name))
            var.set_value(tensor, scope)

        self._init_params(main_program, scope, send_ctx, dense_map)

    def _save_one_table(self, table_id, path, mode):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._worker.save_one_model(table_id, path, mode)
        fleet.util.barrier()

    def _save_dense_params(self, *args, **kwargs):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._ps_save_dense_params(*args, **kwargs)
        fleet.util.barrier()

    def _save_persistables(self, *args, **kwargs):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._save_distributed_persistables(*args, **kwargs)
        fleet.util.barrier()

    def _save_inference_model(self, *args, **kwargs):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._ps_inference_save_inference_model(*args, **kwargs)
        fleet.util.barrier()

    def _load_one_table(self, table_id, path, mode):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._worker.load_one_table(table_id, path, mode)
        fleet.util.barrier()

    def _load_persistables(self, path, mode):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._worker.load_model(path, mode)
        fleet.util.barrier()

    def _load_inference_model(self, path, mode):
        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            self._ps_inference_load_inference_model(path, mode)
        fleet.util.barrier()

    def _shrink(self, threshold=None):
        if threshold is not None:
            warnings.warn(
                "The param threshold is not used in MemorySparseTable, if you need to shrink, please set the config of accessor"
            )
        else:
            threshold = 0

        fleet.util.barrier()
        if self.role_maker._is_first_worker():
            sparses = get_the_one_recv_context(
                self.context,
                is_dense=False,
                split_dense_table=self.role_maker._is_heter_parameter_server_mode,
            )

            for id, names in sparses.items():
                self._worker.shrink_sparse_table(id, threshold)
        fleet.util.barrier()

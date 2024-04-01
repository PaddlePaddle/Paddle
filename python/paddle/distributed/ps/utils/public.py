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

import collections
import logging
import os
import warnings
from functools import reduce

from paddle.base.framework import generate_control_dev_var_name
from paddle.distributed.io import is_persistable
from paddle.framework import core

# logging.basicConfig(
#    format='%(levelname)s - %(asctime)s - %(pathname)s: %(lineno)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger(__name__)

OP_NAME_SCOPE = "op_namescope"
CLIP_OP_NAME_SCOPE = "gradient_clip"
STEP_COUNTER = "@PS_STEP_COUNTER@"
LEARNING_RATE_DECAY_COUNTER = "@LR_DECAY_COUNTER@"

OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleAttrName()
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
op_role = core.op_proto_and_checker_maker.OpRole
op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
backward = core.op_proto_and_checker_maker.OpRole.Backward
OP_DEVICE_KEY = core.op_proto_and_checker_maker.kOpDeviceAttrName()

DEVICE_LIST = ["cpu", "gpu", "xpu"]
COMMUNICATE_OPS_TYPE = ["send", "recv", "fetch_barrier", "send_barrier"]
SPARSE_OP_LIST = ["lookup_table", "lookup_table_v2"]
SPARSE_OP_TYPE_DICT = {"lookup_table": "W", "lookup_table_v2": "W"}
SPARSE_GRAD_OP_TYPE_DICT = {
    "lookup_table_grad": "W",
    "lookup_table_v2_grad": "W",
}
DEFAULT_DEVICE = 'cpu'

DATA_NORM_NAME = [".batch_size", ".batch_sum", ".batch_square_sum"]
DATA_NORM_GRAD_NAME = [x + "@GRAD" for x in DATA_NORM_NAME]


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.WARNING)
    handler = logging.FileHandler(
        log_path, mode='a', encoding='UTF-8', delay=True
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)s - %(asctime)s - %(pathname)s: %(lineno)s - %(message)s'
    )
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


ps_log_root_dir = './ps_log/'
logger = logger_config(
    log_path='./ps_usr_print_log', logging_name='ps_usr_print_log'
)


class DistributedMode:
    SYNC = 0
    ASYNC = 1
    HALF_ASYNC = 2
    GEO = 3
    FL = 4
    NU = 5


class TrainerRuntimeConfig:
    def __init__(self, valid_strategy):
        self.mode = None
        num_threads = os.getenv("CPU_NUM", "1")
        send_queue_size = num_threads
        k_steps = valid_strategy.a_sync_configs["k_steps"]

        if not valid_strategy.a_sync and k_steps == 0:
            self.mode = DistributedMode.SYNC

        if valid_strategy.a_sync and k_steps == 0:
            self.mode = DistributedMode.ASYNC

        if valid_strategy.a_sync and k_steps > 0:
            self.mode = DistributedMode.GEO
            send_queue_size = k_steps

        self.runtime_configs = {}
        self.runtime_configs['communicator_max_merge_var_num'] = os.getenv(
            "FLAGS_communicator_max_merge_var_num", send_queue_size
        )
        self.runtime_configs['communicator_send_queue_size'] = os.getenv(
            "FLAGS_communicator_send_queue_size", send_queue_size
        )
        self.runtime_configs[
            'communicator_independent_recv_thread'
        ] = os.getenv("FLAGS_communicator_independent_recv_thread", "1")
        self.runtime_configs[
            'communicator_min_send_grad_num_before_recv'
        ] = os.getenv(
            "FLAGS_communicator_min_send_grad_num_before_recv", num_threads
        )
        self.runtime_configs['communicator_thread_pool_size'] = os.getenv(
            "FLAGS_communicator_thread_pool_size", "5"
        )
        self.runtime_configs['communicator_send_wait_times'] = os.getenv(
            "FLAGS_communicator_send_wait_times", "5"
        )
        self.runtime_configs['communicator_is_sgd_optimizer'] = os.getenv(
            "FLAGS_communicator_is_sgd_optimizer", "1"
        )

    def get_communicator_flags(self):
        need_keys = []
        num_threads = os.getenv("CPU_NUM", "1")
        mode_str = ""
        if self.mode is None or self.mode == DistributedMode.ASYNC:
            need_keys = self.runtime_configs.keys()
            mode_str = "async"
        elif (
            self.mode == DistributedMode.SYNC
            or self.mode == DistributedMode.HALF_ASYNC
        ):
            mode_str = "sync or half_async"
            need_keys = [
                'communicator_max_merge_var_num',
                'communicator_send_wait_times',
                'communicator_thread_pool_size',
                'communicator_send_queue_size',
            ]
        elif self.mode == DistributedMode.GEO:
            mode_str = "GEO"
            need_keys = [
                'communicator_thread_pool_size',
                'communicator_send_wait_times',
                'communicator_max_merge_var_num',
                'communicator_send_queue_size',
            ]
        else:
            raise ValueError("Unsupported Mode")

        if (
            self.mode == DistributedMode.SYNC
            or self.mode == DistributedMode.HALF_ASYNC
        ):
            max_merge_var_num = self.runtime_configs[
                'communicator_max_merge_var_num'
            ]
            send_queue_size = self.runtime_configs[
                'communicator_send_queue_size'
            ]
            if max_merge_var_num != num_threads:
                print(
                    f'WARNING: In {mode_str} mode, communicator_max_merge_var_num '
                    'must be equal to CPU_NUM. But received, '
                    f'communicator_max_merge_var_num = {max_merge_var_num}, CPU_NUM = '
                    f'{num_threads}. communicator_max_merge_var_num will be forced to {num_threads}.'
                )
                self.runtime_configs[
                    'communicator_max_merge_var_num'
                ] = num_threads
            if send_queue_size != num_threads:
                print(
                    f'WARNING: In {mode_str} mode, communicator_send_queue_size '
                    'must be equal to CPU_NUM. But received, '
                    f'communicator_send_queue_size = {send_queue_size}, CPU_NUM = '
                    f'{num_threads}. communicator_send_queue_size will be forced to {num_threads}.'
                )
                self.runtime_configs[
                    'communicator_send_queue_size'
                ] = num_threads

        return {key: str(self.runtime_configs[key]) for key in need_keys}


def get_lr_ops(program):
    lr_ops = []
    for index, op in enumerate(program.global_block().ops):
        role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
        if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or role_id == int(
            LR_SCHED_OP_ROLE_ATTR_VALUE
        ) | int(OPT_OP_ROLE_ATTR_VALUE):
            lr_ops.append(op)
    return lr_ops


def get_optimize_ops(_program, remote_sparse=[]):
    block = _program.global_block()
    opt_ops = []
    for op in block.ops:
        if _is_opt_role_op(op):
            if (
                len(remote_sparse) > 0
                and op.input("Param")[0] not in remote_sparse
            ):  # for fl: only delete remote sparse optimize
                continue
            # delete clip op from opt_ops when run in Parameter Server mode
            if (
                OP_NAME_SCOPE in op.all_attrs()
                and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE)
            ):
                op._set_attr(
                    "op_role",
                    int(core.op_proto_and_checker_maker.OpRole.Backward),
                )
                continue
            opt_ops.append(op)
    return opt_ops


def get_datanorm_ops(_program):
    block = _program.global_block()
    opt_ops = []
    for op in block.ops:
        if op.type == 'data_norm':
            opt_ops.append(op)
    return opt_ops


def get_dist_env():
    trainer_id = int(os.getenv('PADDLE_TRAINER_ID', '0'))
    trainer_endpoints = ''
    current_endpoint = ''
    num_trainers = 0
    if os.getenv('PADDLE_TRAINER_ENDPOINTS'):
        trainer_endpoints = os.getenv('PADDLE_TRAINER_ENDPOINTS')
        current_endpoint = trainer_endpoints.split(',')[trainer_id]
        num_trainers = len(trainer_endpoints.split(','))

    return {
        'trainer_id': trainer_id,
        'num_trainers': num_trainers,
        'current_endpoint': current_endpoint,
        'trainer_endpoints': trainer_endpoints,
    }


def get_role_id(role_maker):
    try:
        return role_maker._role_id()
    except Exception:
        return role_maker.role_id()


def get_ps_endpoint(role_maker):
    try:
        return role_maker._get_pserver_endpoints()[get_role_id(role_maker)]
    except Exception:
        return role_maker.get_pserver_endpoints()[get_role_id(role_maker)]


def get_ps_endpoints(role_maker):
    try:
        return role_maker._get_pserver_endpoints()
    except Exception:
        return role_maker.get_pserver_endpoints()


def get_heter_worker_endpoint(role_maker):
    return role_maker._get_heter_worker_endpoint()


def get_trainer_endpoint(role_maker):
    return role_maker._get_trainer_endpoint()


def get_trainer_endpoints(role_maker):
    return role_maker._get_trainer_endpoints()


def get_previous_stage_trainers(role_maker):
    try:
        return role_maker._get_previous_trainers()
    except Exception:
        return role_maker.get_previous_trainers()


def is_distributed_sparse_op(op):
    if op.type in SPARSE_OP_LIST and op.attr('is_distributed') is True:
        return True

    if (
        op.type == "distributed_lookup_table"
        and op.attr('is_distributed') is True
    ):
        return True

    return False


def get_sparse_tablename(op):
    return op.input("W")[0]


def is_sparse_op(op):
    if (
        op.type in SPARSE_OP_LIST
        and op.attr('is_sparse') is True
        and op.attr('is_distributed') is False
    ):
        return True

    if (
        op.type == "distributed_lookup_table"
        and op.attr('is_distributed') is False
    ):
        return True

    return False


def get_sparse_tablenames(programs, is_distributed):
    tablenames = set()
    for program in programs:
        if is_distributed:
            for op in program.global_block().ops:
                if is_distributed_sparse_op(op):
                    tablenames.add(get_sparse_tablename(op))
        else:
            for op in program.global_block().ops:
                if is_sparse_op(op):
                    tablenames.add(get_sparse_tablename(op))
    return list(tablenames)


def get_trainers(role_maker):
    try:
        return role_maker._worker_num()
    except Exception:
        return role_maker.worker_num()


def get_dense_send_context(
    program,
    send_ctx,
    idx,
    merged_dense_pairs,
    trainer_id,
    split_dense_table=False,
):
    if len(merged_dense_pairs) < 1:
        return idx
    if not split_dense_table:
        dense_pairs = []
        data_norm_pairs = []
        for merged in merged_dense_pairs:
            is_data_norm = False
            grad = merged[1]
            varname = grad.merged_var.name
            for name in DATA_NORM_GRAD_NAME:
                if varname.endswith(name):
                    is_data_norm = True
            if is_data_norm:
                data_norm_pairs.append(merged)
            else:
                dense_pairs.append(merged)

        # simple dense table
        origin_varnames = []
        var_numel = 0
        for merged in dense_pairs:
            grad = merged[1]
            origin_varnames.append(grad.merged_var.name)
            var = program.global_block().vars[grad.merged_var.name]
            var_numel += reduce(lambda x, y: x * y, var.shape, 1)
        grad_name = "Dense@GRAD_" + str(idx)
        aggregate = True
        # print("public get_dense_send_context dense_table:", grad_name,
        #      var_numel, origin_varnames)
        from paddle.base.core import CommContext

        dense_ctx = CommContext(
            grad_name,
            [grad_name],
            ["127.0.0.1:6071"],
            [var_numel],
            origin_varnames,
            trainer_id,
            aggregate,
            False,
            False,
            idx,
            False,
            False,
            id(program),
            [],
        )
        send_ctx[grad_name] = dense_ctx
        idx += 1

        if len(data_norm_pairs) <= 0:
            return idx

        # data norm table
        origin_varnames = []
        var_numel = 0
        for merged in data_norm_pairs:
            grad = merged[1]
            origin_varnames.append(grad.merged_var.name)
            var = program.global_block().vars[grad.merged_var.name]
            var_numel += reduce(lambda x, y: x * y, var.shape, 1)
        grad_name = "DataNorm@GRAD_" + str(idx)
        aggregate = True
        # print("public get_dense_send_context data_norm table:", grad_name,
        #      var_numel, origin_varnames)
        from paddle.base.core import CommContext

        data_norm_ctx = CommContext(
            grad_name,
            [grad_name],
            ["127.0.0.1:6071"],
            [var_numel],
            origin_varnames,
            trainer_id,
            aggregate,
            False,
            False,
            idx,
            False,
            True,
            id(program),
            [],
        )
        send_ctx[grad_name] = data_norm_ctx
        idx += 1
    else:
        for merged in merged_dense_pairs:
            grad = merged[1]
            origin_varname = grad.merged_var.name
            var = program.global_block().vars[origin_varname]
            var_numel = reduce(lambda x, y: x * y, var.shape, 1)
            grad_name = origin_varname
            aggregate = True
            from paddle.base.core import CommContext

            dense_ctx = CommContext(
                grad_name,
                [grad_name],
                ["127.0.0.1:6071"],
                [var_numel],
                [origin_varname],
                trainer_id,
                aggregate,
                False,
                False,
                idx,
                False,
                False,
                id(program),
                [],
            )
            send_ctx[grad_name] = dense_ctx
            idx += 1
    return idx


def get_geo_trainer_send_context(attrs):
    if attrs['ps_mode'] != DistributedMode.GEO:
        raise ValueError(
            "ps mode: {} not matched {}",
            format(attrs['ps_mode'], "get_geo_trainer_send_context"),
        )
    send_ctx = {}
    trainer_id = get_role_id(attrs['role_maker'])
    origin_programs = attrs['origin_main_programs']
    idx = 0  # table idx

    distributed_varnames = get_sparse_tablenames(origin_programs, True)
    for i, program in enumerate(origin_programs):
        merged_sparse_pairs = attrs['merged_sparse_pairs'][i]
        for merged in merged_sparse_pairs:
            param, grad = merged
            grad_name = grad.merged_var.name
            param_name = param.merged_var.name
            if param_name in attrs['remote_sparse']:  # for recall/ncf model
                continue

            is_distributed = (
                True if param_name in distributed_varnames else False
            )
            var = program.global_block().vars[grad.merged_var.name]
            var_numel = reduce(lambda x, y: x * y, var.shape[1:], 1)
            from paddle.base.core import CommContext

            print(
                "public get_the_geo_send_context sparse: ", grad_name, var_numel
            )
            sparse_ctx = CommContext(
                grad_name,
                [grad_name],
                ["127.0.0.1:6071"],
                [var_numel],
                [grad_name],
                trainer_id,
                True,
                True,
                is_distributed,
                idx,
                False,
                False,
                id(program),
                [],
            )
            idx += 1
            send_ctx[sparse_ctx.var_name()] = sparse_ctx

    if len(send_ctx) == 0:
        raise ValueError("GeoSGD require sparse parameters in your net.")

    if len(attrs['tensor_table']) > 0 and attrs['is_worker']:
        name, ctx = _step_ctx(idx, attrs['role_maker'])
        send_ctx[name] = ctx

    return send_ctx


def _step_ctx(idx, role_maker):
    name = STEP_COUNTER
    trainer_id = get_role_id(role_maker)
    endpoints = get_ps_endpoints(role_maker)
    sections = [1] * len(endpoints)
    names = [name] * len(endpoints)
    from paddle.base.core import CommContext

    ctx = CommContext(
        name,
        names,
        endpoints,
        sections,
        [name],
        trainer_id,
        True,
        False,
        False,
        idx,
        True,
        False,
        -1,
        [],
    )
    return name, ctx


def get_the_one_send_context(attrs, split_dense_table=False, ep_list=None):
    if ep_list is None:
        ep_list = ["127.0.0.1:6071"]
    send_ctx = {}
    trainer_id = get_role_id(attrs['role_maker'])
    origin_programs = attrs['origin_main_programs']
    print(f"is_heter_ps_mode? {split_dense_table}")

    idx = 0
    distributed_varnames = get_sparse_tablenames(origin_programs, True)
    # print("public distributed_varnames:", distributed_varnames)
    for i, program in enumerate(origin_programs):
        merged_sparse_pairs = attrs['merged_sparse_pairs'][i]
        for merged in merged_sparse_pairs:
            param, grad = merged
            grad_name = grad.merged_var.name
            param_name = param.merged_var.name

            remote_sparse_ids = []
            if param_name in attrs['remote_sparse']:  # for recall/ncf model
                remote_sparse_ids.append(idx)

            splited_varname = []
            for i in range(len(ep_list)):
                splited_varname.append(f"{param_name}.block{i}")

            is_distributed = (
                True if param_name in distributed_varnames else False
            )

            var = program.global_block().vars[grad.merged_var.name]

            shape = list(var.shape)
            shape[0] = 0 if is_distributed else shape[0]

            if grad_name in send_ctx:
                continue
            from paddle.base.core import CommContext

            print(
                "public get_the_one_send_context sparse: ",
                grad_name,
                splited_varname,
                shape,
            )
            sparse_ctx = CommContext(
                grad_name,
                splited_varname,
                ep_list,
                shape,
                [grad_name],
                trainer_id,
                True,
                True,
                is_distributed,
                idx,
                False,
                False,
                id(program),
                remote_sparse_ids,
            )

            idx += 1
            send_ctx[sparse_ctx.var_name()] = sparse_ctx

    for i, program in enumerate(origin_programs):
        merged_dense_pairs = attrs['merged_dense_pairs'][i]
        idx = get_dense_send_context(
            program,
            send_ctx,
            idx,
            merged_dense_pairs,
            trainer_id,
            split_dense_table,
        )

    if len(attrs['tensor_table']) > 0 and attrs['is_worker']:
        name, ctx = _step_ctx(idx, attrs['role_maker'])
        send_ctx[name] = ctx

    return send_ctx


def find_heter_ops(program, default_device="cpu"):
    if default_device not in DEVICE_LIST:
        raise ValueError(
            f"Given device {default_device} is not in device list {DEVICE_LIST}"
        )

    def _is_heter_op(op, current_heter_device, default_device="cpu"):
        heter_devices = list(DEVICE_LIST)
        heter_devices.remove(default_device)
        op_device = op.attr("op_device")
        op_type = op.type
        if op_device in heter_devices:
            return True
        elif (
            op_type in COMMUNICATE_OPS_TYPE
            and current_heter_device != default_device
        ):
            # for distributed communicate ops: send & recv & barrier etc.
            # Todo: need update this method
            # op._set_attr('op_device', current_heter_device)
            return True
        elif op_device is None or op_device == default_device:
            op._set_attr('op_device', default_device)
            return False
        return False

    def _is_same_device(op, pre_device, default_device="cpu"):
        op_device = op.attr("op_device")
        if op_device == pre_device:
            return True
        if pre_device == default_device:
            return True
        return False

    def _append_heter_op(op, current_heter_block_ops, heter_ops):
        op_device = op.attr("op_device")
        if op_device not in heter_ops:
            heter_ops[op_device] = {}
        current_heter_block_ops.append(op)

    origin_program = program.clone()
    block = program.global_block()
    '''
       re-place sum op to fix bug for union forward backward op
    '''
    var2idx = {}
    op_list = list(block.ops)
    op_size = len(op_list)

    for i in range(op_size - 1, -1, -1):
        op_list = list(block.ops)
        op = op_list[i]
        if "_grad" in op.type:
            forward_op_type = op.type.split("_grad")[0]
            if (
                forward_op_type in SPARSE_OP_TYPE_DICT.keys()
                and op.attr('remote_prefetch') is True
            ):
                param_name = op.input(SPARSE_OP_TYPE_DICT[forward_op_type])[0]
                if param_name in var2idx:
                    # insert sum op & remove sum op from var2idx and origin place
                    op_list = list(block.ops)
                    sum_op = op_list[var2idx[param_name]]
                    sum_op_inputs = {
                        sum_op.input_names[0]: [
                            block.vars[input]
                            for input in sum_op.input_arg_names
                        ]
                    }
                    sum_op_outputs = {
                        sum_op.output_names[0]: [
                            block.vars[output]
                            for output in sum_op.output_arg_names
                        ]
                    }
                    block._insert_op(
                        index=i + 1,
                        type=sum_op.type,
                        inputs=sum_op_inputs,
                        outputs=sum_op_outputs,
                        attrs=sum_op.all_attrs(),
                    )
                    block._remove_op(var2idx[param_name] + 1)
                    var2idx.pop(param_name)
                    for var_ in var2idx:
                        var2idx[var_] += 1
            elif forward_op_type == "elementwise_mul":
                """
                get output varname of pre op

                """
                output_vars_no_grad = []
                for key in op.output_names:
                    for varname in op.output(key):
                        if varname == "@EMPTY@":
                            continue
                        if "lod_tensor_blocking_queue" in varname:
                            continue
                        output_vars_no_grad.append(varname.split("@GRAD")[0])
                for no_grad_var in output_vars_no_grad:
                    if no_grad_var in var2idx:
                        """
                        insert sum op & remove sum op from var2idx and origin place

                        """
                        op_list = list(block.ops)
                        sum_op = op_list[var2idx[no_grad_var]]
                        sum_op_inputs = {
                            sum_op.input_names[0]: [
                                block.vars[input]
                                for input in sum_op.input_arg_names
                            ]
                        }
                        sum_op_outputs = {
                            sum_op.output_names[0]: [
                                block.vars[output]
                                for output in sum_op.output_arg_names
                            ]
                        }
                        block._insert_op(
                            index=i + 1,
                            type=sum_op.type,
                            inputs=sum_op_inputs,
                            outputs=sum_op_outputs,
                            attrs=sum_op.all_attrs(),
                        )
                        block._remove_op(var2idx[no_grad_var] + 1)
                        var2idx.pop(no_grad_var)
                        for var_ in var2idx:
                            var2idx[var_] += 1
        else:
            if op.type == "sum":
                var = op.output("Out")[0]
                if "@GRAD" in var:
                    origin_var = var.split("@GRAD")[0]
                    pre_op = op_list[i - 1]
                    if "_grad" in pre_op.type:
                        forward_op_type = pre_op.type.split("_grad")[0]
                        if (
                            forward_op_type in SPARSE_OP_TYPE_DICT.keys()
                            and pre_op.attr('remote_prefetch') is True
                        ):
                            param_name = pre_op.input(
                                SPARSE_OP_TYPE_DICT[forward_op_type]
                            )[0]
                            if param_name == origin_var and op.attr(
                                "op_device"
                            ) == pre_op.attr("op_device"):
                                continue
                            else:
                                var2idx[origin_var] = i
                        elif forward_op_type == "elementwise_mul":
                            output_vars = []
                            for key in pre_op.output_names:
                                for varname in pre_op.output(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    output_vars.append(varname)
                            input_vars = []
                            for key in op.input_names:
                                for varname in op.input(key):
                                    if varname == "@EMPTY@":
                                        continue
                                    if "lod_tensor_blocking_queue" in varname:
                                        continue
                                    input_vars.append(varname)
                            is_match = False
                            for varname in output_vars:
                                if varname in input_vars:
                                    is_match = True
                                    break
                            if is_match:
                                continue
                            else:
                                var2idx[origin_var] = i
                    else:
                        var2idx[origin_var] = i

    origin_program = program.clone()
    block = program.global_block()

    program_block_ops = []
    default_ops = {default_device: {}}
    heter_ops = {}
    block_index = 0

    current_heter_block_ops = []
    current_default_block_ops = []
    current_heter_device = default_device
    is_heter = False
    for op in block.ops:
        if _is_heter_op(op, current_heter_device, default_device):
            # for gpu/xpu-op
            is_heter = True

            # for cpu-op block append
            if len(current_default_block_ops) > 1:
                default_ops[default_device][
                    block_index
                ] = current_default_block_ops
                program_block_ops.append(current_default_block_ops)
                current_default_block_ops = []
                block_index += 1

            if _is_same_device(op, current_heter_device, default_device):
                # for gpu-op, gpu-op -> gpu-op,...
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)
            else:
                # for gpu-op -> xpu-op, ...
                op_device = current_heter_block_ops[0].attr("op_device")
                heter_ops[op_device][block_index] = current_heter_block_ops
                program_block_ops.append(current_heter_block_ops)
                block_index += 1
                current_heter_block_ops = []
                current_heter_device = op.attr("op_device")
                _append_heter_op(op, current_heter_block_ops, heter_ops)

        elif is_heter:
            # for gpu/xpu-op -> cpu-op
            op_device = current_heter_block_ops[0].attr("op_device")
            heter_ops[op_device][block_index] = current_heter_block_ops
            program_block_ops.append(current_heter_block_ops)
            block_index += 1
            current_heter_block_ops = []
            current_heter_device = default_device
            is_heter = False
            current_default_block_ops.append(op)
        else:
            # for cpu-op
            current_default_block_ops.append(op)

    if current_default_block_ops != []:
        default_ops[default_device][block_index] = current_default_block_ops
        program_block_ops.append(current_default_block_ops)

    if current_heter_block_ops != []:
        op_device = current_heter_block_ops[0].attr("op_device")
        heter_ops[op_device][block_index] = current_heter_block_ops
        program_block_ops.append(current_heter_block_ops)

    if len(heter_ops) == 0:
        warnings.warn(
            "No heterogeneous OP was found in your program , "
            " please using static.device_guard() to run OPs on different device."
        )

    total_heter_ops = 0
    heter_blocks = 0
    for device in heter_ops.keys():
        heter_block_dict = heter_ops[device]
        heter_blocks += len(heter_block_dict)
        for _, heter_block in heter_block_dict.items():
            total_heter_ops += len(heter_block)
    print(
        f"There are {len(block.ops)} OPs in your main_program, and contains {total_heter_ops} heter-OPs which is made up of {heter_blocks} heter-blocks."
    )

    return origin_program, heter_ops, default_ops, program_block_ops


def union_forward_gradient_op(program_block_ops_list):
    """
    before analyzing the input & output of each block in program_block_list, we should
    union the forward op and corresponding gradient op to eliminate the unnecessary variable
    transmit
    """
    """
    fix for 2emb model, re-place sum op

    """
    block_length = len(program_block_ops_list)
    union_program_block_ops_list = []
    assert (
        block_length % 2 != 0
    ), "the length of program_block_ops_list should be odd"
    for i in range(0, block_length // 2):
        block_op_list = {"forward": program_block_ops_list[i]}
        block_op_list.update(
            {"backward": program_block_ops_list[block_length - 1 - i]}
        )
        union_program_block_ops_list.append(block_op_list)

    block_op_list = {"forward": [], "backward": []}
    for op in program_block_ops_list[block_length // 2]:
        if "_grad" not in op.type and not (op.type == "sum"):
            block_op_list["forward"].append(op)
        else:
            block_op_list["backward"].append(op)
    union_program_block_ops_list.append(block_op_list)
    return union_program_block_ops_list


def find_block_joints(program, program_block_ops_list, heter_ops):
    block_var_detail = find_entrance_exit_private(
        program, program_block_ops_list
    )
    block_var_detail = entrance_exit_check(
        program, program_block_ops_list, block_var_detail, heter_ops
    )
    block_var_detail = delete_block_useless_exit(
        program, program_block_ops_list, block_var_detail
    )

    return block_var_detail


def find_ops_list_input_output(program, ops_list):
    input_var_list = []
    output_var_list = []
    for op in ops_list:
        inputs = _get_input_map_from_op(program.global_block().vars, op)
        input_var_list += get_varlist_from_op_map(inputs)
        outputs = _get_output_map_from_op(program.global_block().vars, op)
        output_var_list += get_varlist_from_op_map(outputs)

    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def find_entrance_exit_private(program, program_block_ops_list):
    block_var_detail = []
    persistables = []
    for index, block_op_list in enumerate(program_block_ops_list):
        # forward
        block_input, block_output = find_ops_list_input_output(
            program, block_op_list["forward"]
        )
        persistables = screen_persistables(
            program, block_input
        ) + screen_persistables(program, block_output)
        # find entrance & exit
        block_private_vars = list(set(block_input) & set(block_output))
        block_entrance = list(set(block_input) - set(block_private_vars))
        block_exit = list(set(block_output) - set(block_private_vars))
        detail = {
            "forward": {
                "entrance": block_entrance,
                "exit": block_exit,
                "private": block_private_vars,
                "persistables": persistables,
            }
        }

        # backward
        bp_block_input, bp_block_output = find_ops_list_input_output(
            program, block_op_list["backward"]
        )
        bp_persistables = screen_persistables(
            program, bp_block_input
        ) + screen_persistables(program, bp_block_output)
        # find entrance & exit
        bp_block_private_vars = list(set(bp_block_input) & set(bp_block_output))
        bp_block_entrance = list(
            set(bp_block_input) - set(bp_block_private_vars)
        )
        bp_block_exit = list(set(bp_block_output) - set(bp_block_private_vars))
        detail.update(
            {
                "backward": {
                    "entrance": bp_block_entrance,
                    "exit": bp_block_exit,
                    "private": bp_block_private_vars,
                    "persistables": bp_persistables,
                }
            }
        )
        block_var_detail.append(detail)
    return block_var_detail


def entrance_exit_check(
    program, program_block_ops_list, block_var_detail, heter_ops
):
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        previous_block_exit = block_var_detail[index - 1]["forward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["forward"]["entrance"]

        backward_entrance = block_var_detail[index]["backward"]["entrance"]

        forward_all = (
            block_var_detail[index]["forward"]["entrance"]
            + block_var_detail[index]["forward"]["private"]
            + block_var_detail[index]["forward"]["exit"]
        )

        for var in backward_entrance:
            if "@GRAD" not in var and var not in forward_all:
                current_block_entrance.append(var)

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance)
        )
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        # var in different stage should not be ignored, since they are not placed in the same program & device
        # need_add_vars = find_need_var_from_previous_block(
        #    need_add_vars, block_var_detail, index, heter_ops)

        previous_block_private = block_var_detail[index - 1]["forward"][
            "private"
        ]
        previous_block_entrance = block_var_detail[index - 1]["forward"][
            "entrance"
        ]
        for var in need_add_vars:
            if (
                var not in previous_block_private
                and var not in previous_block_entrance
            ):
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
            if var not in current_block_entrance:
                current_block_entrance.append(var)

    for index in range(0, len(block_var_detail) - 1, 1):
        previous_block_exit = block_var_detail[index + 1]["backward"]["exit"]
        previous_block_exit.sort()
        current_block_entrance = block_var_detail[index]["backward"]["entrance"]

        current_block_entrance.sort()

        if previous_block_exit == current_block_entrance:
            continue
        exist_vars = list(
            set(previous_block_exit) & set(current_block_entrance)
        )
        need_add_vars = list(set(current_block_entrance) - set(exist_vars))
        need_ignore_vars = []
        for var in need_add_vars:
            if "@GRAD" not in var:
                need_ignore_vars.append(var)
        need_add_vars = list(
            set(need_add_vars).difference(set(need_ignore_vars))
        )
        previous_block_private = block_var_detail[index + 1]["backward"][
            "private"
        ]
        previous_block_entrance = block_var_detail[index + 1]["backward"][
            "entrance"
        ]
        for var in need_add_vars:
            if (
                var not in previous_block_private
                and var not in previous_block_entrance
            ):
                previous_block_entrance.append(var)
            previous_block_exit.append(var)
    return block_var_detail


def delete_block_useless_exit(
    program, program_block_ops_list, block_var_detail
):
    # forward
    for index in range(len(block_var_detail)):
        if index == len(block_var_detail) - 1:
            break
        current_block_exit = block_var_detail[index]["forward"]["exit"]
        next_block_entrance = block_var_detail[index + 1]["forward"]["entrance"]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)

        for var in need_delete_var:
            current_block_exit.remove(var)
    # backward
    for index in range(len(block_var_detail) - 1, -1, -1):
        if index - 1 < 0:
            break
        current_block_exit = block_var_detail[index]["backward"]["exit"]
        next_block_entrance = block_var_detail[index - 1]["backward"][
            "entrance"
        ]
        need_delete_var = []
        for var in current_block_exit:
            if var not in next_block_entrance:
                need_delete_var.append(var)
        for var in need_delete_var:
            current_block_exit.remove(var)

    return block_var_detail


def get_communicate_var_info(
    program, block_index, entrance_var_list, type="forward"
):
    input_var_reshape_dim = []
    input_var_reshape_name = []

    if type == "forward":
        block_input_var_name = (
            f"forward_joint_{block_index - 1}_{block_index}@Heter"
        )
    else:
        block_input_var_name = (
            f"backward_joint_{block_index + 1}_{block_index}@Heter"
        )

    entrance_var_list.sort()
    # input
    # Heter_SERVER_BLOCK_index@JOINT_VAR -> slice -> var@Heter_SERVER_BLOCK@INPUT_RESHAPE_VAR -> reshape -> var
    for name in entrance_var_list:
        var = program.global_block().vars[name]
        shape = var.shape
        recv_var_dim = -1 * reduce(lambda x, y: x * y, shape, 1)
        input_var_reshape_dim.append(recv_var_dim)
        input_var_reshape_name.append(f"{name}.input_reshape@Heter")

    info = {
        "input_var_reshape_dim": input_var_reshape_dim,
        "input_var_reshape_name": input_var_reshape_name,
        "block_input_var_name": block_input_var_name,
    }

    return info


def add_vars_by_var_list(var_name_list, origin_program, program, block):
    for var_name in var_name_list:
        if (
            var_name not in program.global_block().vars
            and var_name not in block.vars
        ):
            var = origin_program.global_block().vars[var_name]
            if var.persistable:
                program.global_block()._clone_variable(
                    var, force_persistable=False
                )
            else:
                block._clone_variable(var, force_persistable=False)


def _get_output_map_from_op(varmap, op):
    """Returns a dict from op output name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.output_names:
        vars = []
        for varname in op.output(key):
            if varname == "@EMPTY@":
                continue
            if "lod_tensor_blocking_queue" in varname:
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def get_varlist_from_op_map(var_map):
    var_list = []
    for key, varlist in var_map.items():
        if not isinstance(varlist, list):
            varlist = [varlist]
        for i in range(len(varlist)):
            var = varlist[i]
            var_list.append(var.name)
    return var_list


def _get_input_map_from_op(varmap, op):
    """Returns a dict from op input name to the vars in varmap."""
    iomap = collections.OrderedDict()
    for key in op.input_names:
        vars = []
        for varname in op.input(key):
            if varname == "@EMPTY@":
                continue
            if "lod_tensor_blocking_queue" in varname:
                continue
            vars.append(varmap[varname])
        if len(vars) == 1:
            iomap[key] = vars[0]
        else:
            iomap[key] = vars
    return iomap


def screen_persistables(program, var_list):
    need_remove = []
    for var_name in var_list:
        if "@GRAD" in var_name:
            if "GRAD" != var_name.split("@")[-1]:
                continue
            origin_var_name = var_name.split("@GRAD")[0]
            var = program.global_block().vars[origin_var_name]
        else:
            var = program.global_block().vars[var_name]

        if is_persistable(var):
            need_remove.append(var_name)

    for var_name in need_remove:
        var_list.remove(var_name)
    return need_remove


def block_append_op(program, origin_program, block, op):
    merge_ordereddict = origin_program.global_block().vars.copy()
    merge_ordereddict.update(block.vars)
    inputs = _get_input_map_from_op(merge_ordereddict, op)
    for key, varlist in inputs.items():
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if (
                var.name not in program.global_block().vars
                and var.name not in block.vars
            ):
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False
                    )
                else:
                    block._clone_variable(var, force_persistable=False)

    outputs = _get_output_map_from_op(origin_program.global_block().vars, op)
    for key, varlist in outputs.items():
        if not isinstance(varlist, list):
            varlist = [varlist]
        for var in varlist:
            if (
                var.name not in program.global_block().vars
                and var.name not in block.vars
            ):
                if var.persistable:
                    program.global_block()._clone_variable(
                        var, force_persistable=False
                    )
                else:
                    block._clone_variable(var, force_persistable=False)

    if "_grad" not in op.type:
        # for forward op
        return block.append_op(
            type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs()
        )
    else:
        # for grad op
        op_desc = op.desc
        backward = core.op_proto_and_checker_maker.OpRole.Backward
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()

        # append grad op
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc._set_attr(RPC_OP_ROLE_ATTR_NAME, backward)

        # set device gard
        if op.desc.has_attr(device_attr_name):
            op_device = op_desc.attr(device_attr_name)
            new_op_desc._set_attr(device_attr_name, op_device)
        block._sync_with_cpp()


def get_next_stage_trainers(role_maker):
    try:
        return role_maker._get_next_trainers()
    except Exception:
        return role_maker.get_next_trainers()


def insert_communicate_op(
    origin_program,
    role_maker,
    heter_block,
    stage_id,
    first_op_index,
    block_var_detail,
    device,
    is_forward=True,
):
    if is_forward:
        next_heter_worker_endpoints = get_next_stage_trainers(role_maker)
        previous_heter_worker_endpoints = get_previous_stage_trainers(
            role_maker
        )
        entrance_var = block_var_detail[stage_id]["forward"]["entrance"]
        comm_info = get_communicate_var_info(
            origin_program, stage_id + 1, entrance_var
        )

    else:
        next_heter_worker_endpoints = get_next_stage_trainers(role_maker)
        previous_heter_worker_endpoints = get_previous_stage_trainers(
            role_maker
        )
        entrance_var = block_var_detail[stage_id - 1]["backward"]["exit"]
        comm_info = get_communicate_var_info(
            origin_program, stage_id - 1, entrance_var, "backward"
        )

    heter_block._insert_op(
        index=first_op_index,
        type="send_and_recv",
        inputs={"X": heter_block.vars[entrance_var[0]]},
        outputs={"Out": []},
        attrs={
            "mode": "forward" if is_forward else "backward",
            "send_var_name": entrance_var + ["microbatch_id"],
            "recv_var_name": [],
            "message_name": comm_info["block_input_var_name"],
            "next_endpoints": next_heter_worker_endpoints,
            "previous_endpoints": previous_heter_worker_endpoints,
            "trainer_id": get_role_id(role_maker),
            "op_device": device,
            RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
        },
    )

    return entrance_var


def get_the_one_recv_context(context, is_dense=True, split_dense_table=False):
    recv_id_maps = {}
    grad_name_to_param_name = {}
    if is_dense:
        send_ctx = get_the_one_send_context(
            context, split_dense_table=split_dense_table
        )
        for idx, (name, ctx) in enumerate(send_ctx.items()):
            if ctx.is_sparse():
                continue
            if ctx.is_tensor_table():
                continue

            origin_grad_varnames = ctx.origin_varnames()

            param_names = []
            for grad_varname in origin_grad_varnames:
                param_name = context["grad_name_to_param_name"][grad_varname]
                param_names.append(param_name)
            recv_id_maps[ctx.table_id()] = param_names
    else:
        send_ctx = get_the_one_send_context(
            context, split_dense_table=False, ep_list=None
        )
        for idx, (name, ctx) in enumerate(send_ctx.items()):
            if not ctx.is_sparse():
                continue

            origin_grad_varnames = ctx.origin_varnames()

            param_names = []
            for grad_varname in origin_grad_varnames:
                param_name = context["grad_name_to_param_name"][grad_varname]
                param_names.append(param_name)
            recv_id_maps[ctx.table_id()] = param_names
    return recv_id_maps


def _get_varname_parts(varname):
    # returns origin, blockid, trainerid
    orig_var_name = ""
    trainer_part = ""
    block_part = ""
    trainer_idx = varname.find(".trainer_")
    if trainer_idx >= 0:
        trainer_part = varname[trainer_idx + 1 :]
    else:
        trainer_idx = len(varname)
    block_index = varname.find(".block")
    if block_index >= 0:
        block_part = varname[block_index + 1 : trainer_idx]
    else:
        block_index = len(varname)
    orig_var_name = varname[0 : min(block_index, trainer_idx)]
    return orig_var_name, block_part, trainer_part


dtype_to_size = {
    core.VarDesc.VarType.FP16: 2,
    core.VarDesc.VarType.FP32: 4,
    core.VarDesc.VarType.FP64: 8,
    core.VarDesc.VarType.INT16: 2,
    core.VarDesc.VarType.INT32: 4,
    core.VarDesc.VarType.INT64: 8,
    core.VarDesc.VarType.BOOL: 1,
    core.VarDesc.VarType.UINT8: 1,
}


def get_var_mem_size(var):
    m_size = reduce(lambda x, y: x * y, var.shape, 1)
    m_size *= dtype_to_size[var.dtype]
    return m_size


class MergedVariable:
    def __init__(self, merged, ordered, offsets):
        self.merged_var = merged
        self.ordered_vars = ordered
        self.offsets = offsets


def build_var_distributed(context):
    origin_programs = context['origin_main_programs']

    param_name_to_grad_name = {}
    grad_name_to_param_name = {}
    context["origin_sparse_pairs"] = []
    context["origin_dense_pairs"] = []
    context["merged_sparse_pairs"] = []
    context['merged_dense_pairs'] = []
    context["merged_variables_pairs"] = []
    context["merged_variable_map"] = {}
    for origin_program in origin_programs:
        sparse_pairs, dense_pairs = get_param_grads(origin_program)
        # print("public build_var_distributed sparse_pairs:", sparse_pairs)
        # print("public build_var_distributed dense_pairs:", dense_pairs)
        origin_for_sparse = []
        origin_for_dense = []
        merged_sparse_pairs = []
        merged_dense_pairs = []
        merged_variables_pairs = []

        for param, grad in sparse_pairs:
            origin_for_sparse.append((param, grad))

        for param, grad in dense_pairs:
            origin_for_dense.append((param, grad))

        for dense_pair in origin_for_dense:
            param, grad = dense_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            merged_variables_pairs.append((m_param, m_grad))
            merged_dense_pairs.append((m_param, m_grad))
        # print("public build_var_distributed merged_dense_pairs:",
        #       merged_dense_pairs)

        for sparse_pair in origin_for_sparse:
            param, grad = sparse_pair

            m_param = MergedVariable(param, [param], [0])
            m_grad = MergedVariable(grad, [grad], [0])
            merged_variables_pairs.append((m_param, m_grad))
            merged_sparse_pairs.append((m_param, m_grad))
        # print("public build_var_distributed merged_sparse_pairs:",
        #       merged_sparse_pairs)

        for merged in merged_variables_pairs:
            m_param, m_grad = merged
            context["merged_variable_map"][
                m_param.merged_var.name
            ] = m_param.merged_var
            context["merged_variable_map"][
                m_grad.merged_var.name
            ] = m_grad.merged_var

        param_merges = []
        param_merges.extend(origin_for_sparse)
        param_merges.extend(origin_for_dense)

        for param, grad in param_merges:
            param_name_to_grad_name[param.name] = grad.name
            grad_name_to_param_name[grad.name] = param.name

        context["origin_sparse_pairs"].append(origin_for_sparse)
        context["origin_dense_pairs"].append(origin_for_dense)
        context["merged_sparse_pairs"].append(merged_sparse_pairs)
        context['merged_dense_pairs'].append(merged_dense_pairs)

    context["param_name_to_grad_name"] = param_name_to_grad_name
    context["grad_name_to_param_name"] = grad_name_to_param_name
    '''
    print("public build_var_distributed origin_sparse_pairs:",
        context["origin_sparse_pairs"])
    print("public build_var_distributed origin_for_dense:",
        context["origin_dense_pairs"])
    print("public build_var_distributed merged_sparse_pairs:",
        context["merged_sparse_pairs"])
    print("public build_var_distributed merged_dense_pairs:",
        context['merged_dense_pairs'])
    print("public build_var_distributed param_name_to_grad_name:",
        param_name_to_grad_name)
    print("public build_var_distributed grad_name_to_param_name:",
        grad_name_to_param_name)
    '''


def _is_opt_role_op(op):
    # NOTE : depend on oprole to find out whether this op is for
    # optimize
    op_maker = core.op_proto_and_checker_maker
    optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
    if op_maker.kOpRoleAttrName() in op.attr_names and int(
        op.all_attrs()[op_maker.kOpRoleAttrName()]
    ) == int(optimize_role):
        return True
    return False


def get_param_grads(origin_program):
    def _get_params_grads(sparse_varnames):
        block = origin_program.global_block()

        dense_param_grads = []
        sparse_param_grads = []

        optimize_params = set()
        origin_var_dict = origin_program.global_block().vars
        role_id = int(core.op_proto_and_checker_maker.OpRole.Backward)
        for op in block.ops:
            if _is_opt_role_op(op):
                # delete clip op from opt_ops when run in Parameter Server mode
                if (
                    OP_NAME_SCOPE in op.all_attrs()
                    and CLIP_OP_NAME_SCOPE in op.attr(OP_NAME_SCOPE)
                ):
                    op._set_attr("op_role", role_id)
                    continue
                if not op.has_attr(OP_ROLE_VAR_ATTR_NAME):
                    continue
                if op.attr(OP_ROLE_VAR_ATTR_NAME):
                    param_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
                    grad_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                    if param_name not in optimize_params:
                        optimize_params.add(param_name)
                        param_grad = (
                            origin_var_dict[param_name],
                            origin_var_dict[grad_name],
                        )

                        if param_name in sparse_varnames:
                            sparse_param_grads.append(param_grad)
                        else:
                            dense_param_grads.append(param_grad)
        return sparse_param_grads, dense_param_grads

    def _get_sparse_varnames():
        varnames = []
        for op in origin_program.global_block().ops:
            if (
                op.type in SPARSE_OP_TYPE_DICT.keys()
                and op.attr('remote_prefetch') is True
            ):
                param_name = op.input(SPARSE_OP_TYPE_DICT[op.type])[0]
                varnames.append(param_name)

        return list(set(varnames))

    sparse_varnames = _get_sparse_varnames()
    sparse_param_grads, dense_param_grads = _get_params_grads(sparse_varnames)

    return sparse_param_grads, dense_param_grads


def delete_ops(block, ops):
    for op in ops:
        try:
            idx = list(block.ops).index(op)
            block._remove_op(idx)
        except Exception as e:
            print(e)


def find_send_op(program):
    send_op_list = []
    for op in program.global_block().ops:
        if op.type == "send":
            send_op_list.append(op)
    return send_op_list


def find_op_input_output(program, block, op):
    input_var_list = []
    output_var_list = []
    inputs = _get_input_map_from_op(block.vars, op)
    input_var_list += get_varlist_from_op_map(inputs)
    outputs = _get_output_map_from_op(block.vars, op)
    output_var_list += get_varlist_from_op_map(outputs)
    input_var_list = list(set(input_var_list))
    output_var_list = list(set(output_var_list))
    return input_var_list, output_var_list


def add_send_op(program, block, _vars):
    def _get_send_op_dict():
        send_op_dict = {}
        send_op_list = find_send_op(program)
        for op in send_op_list:
            input_list, _ = find_op_input_output(
                program, program.global_block(), op
            )
            for var in input_list:
                send_op_dict[var] = op
        return send_op_dict

    send_grad_var_list = []
    send_op_dict = _get_send_op_dict()
    table_dict = {}
    for persistable_var in _vars:
        if "@GRAD" not in persistable_var:
            continue
        if "GRAD" != persistable_var.split("@")[-1]:
            continue
        if persistable_var not in send_op_dict:
            continue
        send_op = send_op_dict[persistable_var]
        is_sparse = send_op.attr('is_sparse')
        table_id = send_op.attr('table_id')
        send_varnames = send_op.attr('send_varnames')
        send_grad_var_list.append(persistable_var)
        if table_id not in table_dict:
            table_dict[table_id] = {}
            table_dict[table_id]['var_list'] = []
            table_dict[table_id]['is_sparse'] = is_sparse
            table_dict[table_id]['send_varnames'] = send_varnames
        table_dict[table_id]['var_list'].append(persistable_var)

    for table_id in table_dict:
        dummy_output = block.create_var(name=generate_control_dev_var_name())
        send_input_vars = [
            block.vars[union_var]
            for union_var in table_dict[table_id]['var_list']
        ]
        block.append_op(
            type="send",
            inputs={"X": send_input_vars},
            outputs={"Out": dummy_output},
            attrs={
                "send_varnames": table_dict[table_id]['send_varnames'],
                "is_sparse": is_sparse,
                "table_id": table_id,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
            },
        )

    return send_grad_var_list


def get_vars_name_in_block(block):
    vars_list = block.vars.keys()
    vars_name_list = list(vars_list)
    return vars_name_list


# reserve static_var
def delete_trainer_useless_var(program, static_var):
    static_var = list(set(static_var))
    program_useful_var_list = []
    for op in program.global_block().ops:
        input_var_list, output_var_list = find_op_input_output(
            program, program.global_block(), op
        )
        op_var_list = list(set(input_var_list).union(set(output_var_list)))
        program_useful_var_list = list(
            set(program_useful_var_list).union(set(op_var_list))
        )
    program_useful_var_list += static_var
    program_useless_var_list = list(
        set(get_vars_name_in_block(program.global_block())).difference(
            set(program_useful_var_list)
        )
    )
    for var in program_useless_var_list:
        program.global_block()._remove_var(var)
    return program_useless_var_list


def create_backward_block(
    program, origin_program, bp_ops_list, block_var_detail
):
    pre_block_idx = program.num_blocks - 1
    heter_block = program._create_block(pre_block_idx)

    for _, op in enumerate(bp_ops_list):
        if op.type == "send":
            send_varnames = op.attr('send_varnames')
            is_skip = False
            for varname in send_varnames:
                if (
                    varname not in program.global_block().vars
                    and varname not in heter_block.vars
                ):
                    is_skip = True
                    break
            if is_skip:
                continue
        block_append_op(program, origin_program, heter_block, op)

    entrance_vars = block_var_detail[0]["backward"]["entrance"]
    add_vars_by_var_list(entrance_vars, origin_program, program, heter_block)
    exit_vars = block_var_detail[0]["backward"]["exit"]
    add_vars_by_var_list(exit_vars, origin_program, program, heter_block)
    return heter_block


def is_backward_op(op):
    return op_role_attr_name in op.attr_names and (
        int(op.attr(op_role_attr_name)) & int(op_role.Backward)
    )


def is_forward_op(op):
    return op_role_attr_name in op.attr_names and (
        int(op.attr(op_role_attr_name)) == int(op_role.Forward)
    )


def is_push_sparse_op(op):
    return op.type == 'distributed_push_sparse'


def get_distributed_push_sparse_op_list(block):
    push_sparse_op_list = []
    for op_idx in range(block.desc.op_size()):
        op = block.ops[op_idx]
        if is_push_sparse_op(op):
            push_sparse_op_list.append(op)
    return push_sparse_op_list


def get_bp_op_list(block):
    bp_op_list = []
    for op_idx in range(block.desc.op_size()):
        op = block.ops[op_idx]
        if is_backward_op(op):
            bp_op_list.append(op)
    return bp_op_list


def delete_same_ops(block, ops):
    for op in ops:
        try:
            for origin_op in block.ops:
                if str(origin_op) == str(op):
                    idx = list(block.ops).index(origin_op)
                    block._remove_op(idx)
                    break
        except Exception as e:
            print(e)


def check_program(program):
    block_idx = 0
    for block in program.blocks:
        for op in block.ops:
            input_var_names = op.desc.input_arg_names()
            output_var_names = op.desc.output_arg_names()
            for var_name in input_var_names + output_var_names:
                if not block._find_var_recursive(str(var_name)):
                    raise ValueError(
                        f'var: {str(var_name)} needed by op is not found in block: {block_idx}'
                    )
        block_idx += 1
    print('program checked valid')


def debug_program(file, program):
    # py >= 3.2
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, 'w+') as f:
        f.write(str(program))


def is_distributed_env():
    node_role = os.getenv("TRAINING_ROLE")
    if node_role is None:
        return False
    else:
        return True

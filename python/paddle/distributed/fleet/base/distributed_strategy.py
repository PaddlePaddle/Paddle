# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation. All rights reserved.
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

import copy

import google.protobuf
import google.protobuf.text_format

import paddle
from paddle.base.framework import _global_flags
from paddle.base.wrapped_decorator import wrap_decorator
from paddle.distributed.fleet.proto import distributed_strategy_pb2
from paddle.distributed.fleet.utils.log_util import logger

__all__ = []

non_auto_func_called = True


def __non_auto_func_called__(func):
    def __impl__(*args, **kwargs):
        global non_auto_func_called
        non_auto_func_called = False
        return func(*args, **kwargs)

    return __impl__


is_strict_auto = wrap_decorator(__non_auto_func_called__)


def get_repeated_msg_dict(msg):
    res_list = []
    for item in msg:
        fields = item.DESCRIPTOR.fields
        res_dict = {}
        for f in fields:
            v = getattr(item, f.name)
            if (
                f.label
                == google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED
            ):
                v = list(v)
            res_dict[f.name] = v
        res_list.append(res_dict)
    return res_list


def get_msg_dict(msg):
    res_dict = {}
    fields = msg.DESCRIPTOR.fields
    for f in fields:
        v = getattr(msg, f.name)
        # NOTE(zhiqiu): convert repeated field to list to
        # avoid segment fault when the process exit?
        # WHY?
        # I guess the type or value of protobuf item is NULL when
        # deallocated.
        if f.label == google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED:
            if (
                f.type
                != google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE
            ):
                v = list(v)
            else:
                v = get_repeated_msg_dict(v)
        res_dict[f.name] = v
    return res_dict


def assign_repeated_msg(msg, config):
    for key in config:
        new_item = msg.add()
        fields = new_item.DESCRIPTOR.fields
        for f in fields:
            if key == f.name:
                # LABEL_OPTIONAL = 1
                # LABEL_REPEATED = 3
                # LABEL_REQUIRED = 2
                if f.label == 3:
                    if config[f.name] is not None:
                        new_item = getattr(msg, f.name)
                        if (
                            f.type
                            != google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE
                        ):
                            new_item.extend(config[f.name])
                        else:
                            assign_configs_value(new_item, config[f.name])
                elif f.label == 1 or f.label == 2:
                    setattr(msg, f.name, config[f.name])


def assign_configs_value(msg, config):
    fields = msg.DESCRIPTOR.fields
    for key in config:
        for f in fields:
            if key == f.name:
                # LABEL_OPTIONAL = 1
                # LABEL_REPEATED = 3
                # LABEL_REQUIRED = 2
                if f.label == 3:
                    if config[f.name] is not None:
                        new_item = getattr(msg, f.name)
                        # deal with repeated message
                        if (
                            f.type
                            != google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE
                        ):
                            new_item.extend(config[f.name])
                        else:
                            assign_repeated_msg(new_item, config[f.name])
                elif f.label == 1 or f.label == 2:
                    setattr(msg, f.name, config[f.name])


def check_configs_key(msg, config, field_name):
    key_list = msg.DESCRIPTOR.fields_by_name.keys()
    for key in config:
        assert key in key_list, f"key:{key} not in {field_name}"


class DistributedJobInfo:
    """
    DistributedJobInfo will serialize all distributed training information
    Just for inner use: 1) debug 2) replicate experiments
    """

    def __init__(self):
        self.job_info = distributed_strategy_pb2.DistributedJobInfo()

    def _set_worker_num(self, worker_num):
        self.job_info.worker_num = worker_num

    def _set_server_num(self, server_num):
        self.job_info.server_num = server_num

    def _set_worker_ips(self, worker_ips):
        self.job_info.worker_ips.extend(worker_ips)

    def _set_server_endpoints(self, server_endpoints):
        self.job_info.server_endpoints.extend(server_endpoints)

    def _set_origin_startup(self, origin_startup_prog):
        self.job_info.origin_startup = str(origin_startup_prog)

    def _set_origin_main(self, origin_main_prog):
        self.job_info.origin_main = str(origin_main_prog)

    def _distributed_main(self, distributed_main_prog):
        self.job_info.distributed_main = str(distributed_main_prog)

    def _optimizer_name(self, optimizer_name):
        self.job_info.optimizer_name = optimizer_name

    def _set_distributed_strategy(self, dist_strategy):
        self.job_info.strategy = dist_strategy


ReduceStrategyFleet = int


class DistributedStrategy:
    __lock_attr = False

    def __init__(self):
        """

        DistributedStrategy is the main configuration entry for distributed training of Paddle.
        All of the distributed training configurations can be configured in DistributedStrategy,
        such as automatic mixed precision (AMP), Layer-wise Adaptive Rate Scaling (LARS),
        asynchronous update parameter server(ASGD), etc.

        DistributedStrategy can be serialized into protobuf file or deserialized from protobuf file

        Users who run local training usually configure BuildStrategy and ExecutionStrategy, and
        DistributedStrategy supports configurations from BuildStrategy and ExecutionStrategy

        """
        self.strategy = distributed_strategy_pb2.DistributedStrategy()

        # Set the default values of the following flags to the ones set by users
        key = 'FLAGS_cudnn_batchnorm_spatial_persistent'
        if _global_flags().is_public(key):
            self.strategy.cudnn_batchnorm_spatial_persistent = bool(
                _global_flags()[key]
            )
        key = 'FLAGS_conv_workspace_size_limit'
        if _global_flags().is_public(key):
            self.strategy.conv_workspace_size_limit = int(_global_flags()[key])
        key = 'FLAGS_cudnn_exhaustive_search'
        if _global_flags().is_public(key):
            self.strategy.cudnn_exhaustive_search = bool(_global_flags()[key])
        key = 'FLAGS_sync_nccl_allreduce'
        if _global_flags().is_public(key):
            self.strategy.sync_nccl_allreduce = bool(_global_flags()[key])

        self.hybrid_parallel_order = ['dp', 'pp', 'sharding', 'sep', 'mp']
        self.sync_param_name = ["embedding", "layer_norm", ".b_"]

        self.__lock_attr = True
        logger.info("distributed strategy initialized")

    def __setattr__(self, key, value):
        if self.__lock_attr and not hasattr(self, key):
            raise TypeError(
                f"{key} is not a attribute of {self.__class__.__name__}"
            )
        object.__setattr__(self, key, value)

    def save_to_prototxt(self, output):
        """

        Serialize current DistributedStrategy to string and save to output file

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.dgc = True
                >>> strategy.recompute = True
                >>> strategy.recompute_configs = {"checkpoints": ["x"]}
                >>> strategy.save_to_prototxt("dist_strategy.prototxt")

        """
        with open(output, "w") as fout:
            fout.write(str(self.strategy))

    def load_from_prototxt(self, pb_file):
        """

        Load from prototxt file for DistributedStrategy initialization

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.dgc = True
                >>> strategy.recompute = True
                >>> strategy.recompute_configs = {"checkpoints": ["x"]}
                >>> strategy.save_to_prototxt("dist_strategy.prototxt")

                >>> strategy.load_from_prototxt("dist_strategy.prototxt")

        """
        with open(pb_file, 'r') as f:
            self.strategy = google.protobuf.text_format.Merge(
                str(f.read()), self.strategy
            )

    @property
    def build_strategy(self):
        """

        Configure BuildStrategy for DistributedStrategy
        Note that the properties of BuildStrategy are valid in DistributedStrategy
        only if the property is non-distributed strategy.

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> build_strategy = paddle.static.BuildStrategy()
                >>> build_strategy.enable_sequential_execution = True
                >>> build_strategy.fuse_elewise_add_act_ops = True
                >>> build_strategy.fuse_bn_act_ops = True
                >>> build_strategy.enable_auto_fusion = True
                >>> build_strategy.fuse_relu_depthwise_conv = True
                >>> build_strategy.fuse_broadcast_ops = True
                >>> build_strategy.fuse_all_optimizer_ops = True
                >>> build_strategy.enable_inplace = True

                >>> strategy = paddle.distributed.fleet.DistributedStrategy()
                >>> strategy.build_strategy = build_strategy

        """

        build_strategy = paddle.static.BuildStrategy()
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            value = getattr(self.strategy.build_strategy, f.name)
            if f.name == 'reduce_strategy':
                value = paddle.static.BuildStrategy.ReduceStrategy(value)
            setattr(build_strategy, f.name, value)
        return build_strategy

    @build_strategy.setter
    @is_strict_auto
    def build_strategy(self, strategy):
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            if f.label == 1 or f.label == 2:  # optional and required field
                value = getattr(strategy, f.name)
                if f.name == 'reduce_strategy':
                    value = ReduceStrategyFleet(value)
                setattr(self.strategy.build_strategy, f.name, value)
            elif f.label == 3:  # repeated field
                getattr(self.strategy.build_strategy, f.name).extend(
                    getattr(strategy, f.name)
                )

    @property
    def gradient_scale_configs(self):
        """

        Set the strategy of gradient scale

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.gradient_scale_configs = {'scale_strategy': 'avg'}

        Note that, strategy must be in 'avg', 'sum' or 'customized'

        """
        return get_msg_dict(self.strategy.gradient_scale_configs)

    @gradient_scale_configs.setter
    @is_strict_auto
    def gradient_scale_configs(self, config):
        check_configs_key(
            self.strategy.gradient_scale_configs,
            config,
            'gradient_scale_configs',
        )
        assign_configs_value(self.strategy.gradient_scale_configs, config)

    @property
    def a_sync(self):
        """

        Indicating whether we are using asynchronous stochastic gradient descent updates
        for training. This property is valid when we are using parameter server training,
        which is implied by setting appropriate RoleMaker
        Default value: True

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> role_maker = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role_maker)

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.a_sync = True  # by default this is True

                >>> # code block for defining loss and local optimizer
                >>> # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.a_sync

    @a_sync.setter
    @is_strict_auto
    def a_sync(self, flag):
        if isinstance(flag, bool):
            self.strategy.a_sync = flag
            self.a_sync_configs = {"k_steps": 0}
        else:
            raise ValueError(
                f"The type of `flag` is invalid, expected type is bool, but received {type(flag)}"
            )

    @property
    def a_sync_configs(self):
        """

        Set a_sync update configurations. In general, asynchronous parameter server
        training has several configurable settings that can be configured through
        a dict.

        **Notes**:
            k_step(int): number of local optimization updates before communication

            max_merge_var_num(int): maximum number of merged gradients before communication

            send_queue_size(int): a buffer size of worker communication

            independent_recv_thread(bool): if we are using independent recv thread for communication

            thread_pool_size(int): number of thread pool

            send_wait_times(int): waiting time for sending gradients

            runtime_split_send_recv(bool): if we are using Tensor split for send and recv during runtime

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> role_maker = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role_maker)

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.a_sync = True  # by default this is True
                >>> configs = {"k_steps": 1024, "send_queue_size": 32}
                >>> strategy.a_sync_configs = configs

                >>> # code block for defining loss and local optimizer
                >>> # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return get_msg_dict(self.strategy.a_sync_configs)

    @a_sync_configs.setter
    @is_strict_auto
    def a_sync_configs(self, configs):
        check_configs_key(
            self.strategy.a_sync_configs, configs, "a_sync_configs"
        )
        assign_configs_value(self.strategy.a_sync_configs, configs)

    @property
    def trainer_desc_configs(self):
        """

        Set trainer desc configurations.

        **Notes**:
            dump_fields_path(str): the path of dump fields

            dump_fields(list(str)): the fields that you want to dump

            dump_param(list(str)): the param that you want to dump

            stat_var_names(list(str)):

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> role_maker = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role_maker)

                >>> strategy = fleet.DistributedStrategy()
                >>> configs = {"dump_fields_path": "./dump_data", "dump_fields": ["xxx", "yyy"]}
                >>> strategy.trainer_desc_configs = configs

                >>> # code block for defining loss and local optimizer
                >>> # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return get_msg_dict(self.strategy.trainer_desc_configs)

    @property
    def adam_d2sum(self):
        """

        set adam_d2sum
        Default value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> role_maker = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role_maker)

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.adam_d2sum = True  # by default this is False

                >>> # code block for defining loss and local optimizer
                >>> # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.adam_d2sum

    @adam_d2sum.setter
    @is_strict_auto
    def adam_d2sum(self, flag):
        if isinstance(flag, bool):
            self.strategy.adam_d2sum = flag
        else:
            raise ValueError(
                f"The type of `flag` is invalid, expected type is bool, but received {type(flag)}"
            )

    @trainer_desc_configs.setter
    @is_strict_auto
    def trainer_desc_configs(self, configs):
        check_configs_key(
            self.strategy.trainer_desc_configs, configs, "trainer_desc_configs"
        )
        assign_configs_value(self.strategy.trainer_desc_configs, configs)

    @property
    def fs_client_param(self):
        """

        Set fs client configurations.

        Note:
            uri(str): the uri of fs client

            user(str): the user_name of fs client

            passwd(str): the passwd of fs client

            hadoop_bin(str):

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> role_maker = fleet.PaddleCloudRoleMaker()
                >>> fleet.init(role_maker)
                >>> strategy = fleet.DistributedStrategy()
                >>> configs = {"uri": "xxx", "user": "xxx", "passwd": "xxx"}
                >>> strategy.fs_client_param = configs
                >>> # code block for defining loss and local optimizer
                >>> # sgd = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.fs_client_param

    @fs_client_param.setter
    @is_strict_auto
    def fs_client_param(self, configs):
        check_configs_key(
            self.strategy.fs_client_param, configs, "fs_client_param"
        )
        assign_configs_value(self.strategy.fs_client_param, configs)

    @property
    def sparse_table_configs(self):
        return self.strategy.downpour_table_param

    @sparse_table_configs.setter
    @is_strict_auto
    def sparse_table_configs(self, configs):
        from google.protobuf.descriptor import FieldDescriptor

        table_param = self.strategy.downpour_table_param

        def set_table_config(msg, config_name, configs, index=0):
            for field in msg.DESCRIPTOR.fields:
                name = config_name + "." + field.name
                if field.type == FieldDescriptor.TYPE_MESSAGE:
                    logger.debug(f"message: {name}")
                    if field.label == FieldDescriptor.LABEL_REPEATED:
                        if name + ".num" not in configs:
                            continue
                        num = configs[name + ".num"]
                        logger.debug(f"message num: {name} {num}")
                        for i in range(num):
                            data = getattr(msg, field.name).add()
                            set_table_config(data, name, configs, i)
                    else:
                        set_table_config(
                            getattr(msg, field.name), name, configs
                        )
                else:
                    logger.debug("not message: %s", name)
                    if name not in configs:
                        continue
                    if field.label == FieldDescriptor.LABEL_REPEATED:
                        getattr(msg, field.name).extend(configs[name])
                    else:
                        if type(configs[name]) == list:
                            setattr(msg, field.name, configs[name][index])
                        else:
                            setattr(msg, field.name, configs[name])

        if not configs:
            logger.info("table configs is empty")
        else:
            for table_name in configs:
                table_data = table_param.add()
                table_data.table_name = table_name
                set_table_config(
                    table_data,
                    "table_parameters." + table_name,
                    configs[table_name],
                )

    @sparse_table_configs.setter
    def fleet_desc_configs(self, configs):
        support_sparse_key_list = [
            'sparse_table_class',
            'sparse_compress_in_save',
            'sparse_shard_num',
            'sparse_accessor_class',
            'sparse_learning_rate',
            'sparse_initial_g2sum',
            'sparse_initial_range',
            'sparse_weight_bounds',
            'sparse_fea_dim',
            'sparse_embedx_dim',
            'sparse_embedx_threshold',
            'sparse_nonclk_coeff',
            'sparse_click_coeff',
            'sparse_base_threshold',
            'sparse_delta_threshold',
            'sparse_delta_keep_days',
            'sparse_delete_after_unseen_days',
            'sparse_show_click_decay_rate',
            'sparse_delete_threshold',
            'sparse_converter',
            'sparse_deconverter',
            'sparse_enable_cache',
            'sparse_cache_rate',
            'sparse_cache_file_num',
            'sparse_beta1_decay_rate',
            'sparse_beta2_decay_rate',
            'sparse_ada_epsilon',
            'sparse_optimizer',
            'sparse_ssd_unseenday_threshold',
            'embed_sparse_optimizer',
            'embed_sparse_learning_rate',
            'embed_sparse_weight_bounds',
            'embed_sparse_initial_range',
            'embed_sparse_initial_g2sum',
            'embed_sparse_beta1_decay_rate',
            'embed_sparse_beta2_decay_rate',
            'embedx_sparse_optimizer',
            'embedx_sparse_learning_rate',
            'embedx_sparse_weight_bounds',
            'embedx_sparse_initial_range',
            'embedx_sparse_initial_g2sum',
            'embedx_sparse_beta1_decay_rate',
            'embedx_sparse_beta2_decay_rate',
            'feature_learning_rate',
            'nodeid_slot',
            'sparse_load_filter_slots',
            'sparse_save_filter_slots',
        ]
        support_sparse_table_class = [
            'DownpourSparseTable',
            'DownpourSparseSSDTable',
        ]
        support_sparse_accessor_class = [
            'DownpourSparseValueAccessor',
            'DownpourCtrAccessor',
            'DownpourCtrDoubleAccessor',
            'DownpourUnitAccessor',
            'DownpourDoubleUnitAccessor',
            'DownpourCtrDymfAccessor',
        ]
        table_param = self.strategy.downpour_table_param

        def add_graph_config(graph, strategy):
            graph.feature_learning_rate = strategy.get(
                'feature_learning_rate', 0.05
            )
            graph.nodeid_slot = strategy.get('nodeid_slot', 9008)

        def sparse_optimizer_config(sgd, strategy, prefix):
            optimizer_name = strategy.get(
                prefix + "sparse_optimizer", "adagrad"
            )
            sgd.name = optimizer_name
            if optimizer_name == "naive":
                sgd.name = "SparseNaiveSGDRule"
                sgd.naive.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.05
                )
                sgd.naive.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.naive.weight_bounds.extend(bounds)
            elif optimizer_name == "adagrad":
                sgd.name = 'SparseAdaGradSGDRule'
                sgd.adagrad.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.05
                )
                sgd.adagrad.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                if prefix == "embed_":
                    sgd.adagrad.initial_range = 0
                sgd.adagrad.initial_g2sum = strategy.get(
                    prefix + 'sparse_initial_g2sum', 3
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.adagrad.weight_bounds.extend(bounds)
            elif optimizer_name == "adagrad_v2":
                sgd.name = 'SparseAdaGradV2SGDRule'
                sgd.adagrad.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.05
                )
                sgd.adagrad.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                if prefix == "embed_":
                    sgd.adagrad.initial_range = 0
                sgd.adagrad.initial_g2sum = strategy.get(
                    prefix + 'sparse_initial_g2sum', 3
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.adagrad.weight_bounds.extend(bounds)
            elif optimizer_name == "std_adagrad":
                sgd.name = 'StdAdaGradSGDRule'
                sgd.adagrad.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.05
                )
                sgd.adagrad.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                if prefix == "embed_":
                    sgd.adagrad.initial_range = 0
                sgd.adagrad.initial_g2sum = strategy.get(
                    prefix + 'sparse_initial_g2sum', 3
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.adagrad.weight_bounds.extend(bounds)
            elif optimizer_name == "adam":
                sgd.name = 'SparseAdamSGDRule'
                sgd.adam.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.001
                )
                sgd.adam.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                sgd.adam.beta1_decay_rate = strategy.get(
                    prefix + 'sparse_beta1_decay_rate', 0.9
                )
                sgd.adam.beta2_decay_rate = strategy.get(
                    prefix + 'sparse_beta2_decay_rate', 0.999
                )
                sgd.adam.ada_epsilon = strategy.get(
                    prefix + 'sparse_ada_epsilon', 1e-8
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.adam.weight_bounds.extend(bounds)
            elif optimizer_name == "shared_adam":
                sgd.name = 'SparseSharedAdamSGDRule'
                sgd.adam.learning_rate = strategy.get(
                    prefix + 'sparse_learning_rate', 0.001
                )
                sgd.adam.initial_range = strategy.get(
                    prefix + 'sparse_initial_range', 1e-4
                )
                sgd.adam.beta1_decay_rate = strategy.get(
                    prefix + 'sparse_beta1_decay_rate', 0.9
                )
                sgd.adam.beta2_decay_rate = strategy.get(
                    prefix + 'sparse_beta2_decay_rate', 0.999
                )
                sgd.adam.ada_epsilon = strategy.get(
                    prefix + 'sparse_ada_epsilon', 1e-8
                )
                bounds = strategy.get(
                    prefix + 'sparse_weight_bounds', [-10, 10]
                )
                sgd.adam.weight_bounds.extend(bounds)

        def set_sparse_table_config(table_data, config):
            for key in config:
                if key not in support_sparse_key_list:
                    raise ValueError("strategy key '%s' not support" % (key))
            table_class = config.get(
                "sparse_table_class", "DownpourSparseTable"
            )
            if table_class not in support_sparse_table_class:
                raise ValueError(
                    "support sparse_table_class: ['DownpourSparseTable, DownpourSparseSSDTable'], but actual %s"
                    % (table_class)
                )
            if table_class == "DownpourSparseSSDTable":
                table_data.table_class = 'SSDSparseTable'
            else:
                table_data.table_class = 'MemorySparseTable'
            table_data.shard_num = config.get('sparse_shard_num', 1000)
            table_data.enable_sparse_table_cache = config.get(
                'sparse_enable_cache', True
            )
            table_data.sparse_table_cache_rate = config.get(
                'sparse_cache_rate', 0.00055
            )
            table_data.sparse_table_cache_file_num = config.get(
                'sparse_cache_file_num', 16
            )

            accessor_class = config.get(
                "sparse_accessor_class", "DownpourCtrAccessor"
            )
            if accessor_class not in support_sparse_accessor_class:
                raise ValueError(
                    "support sparse_accessor_class: ['DownpourSparseValueAccessor', 'DownpourCtrAccessor', 'DownpourCtrDoubleAccessor', 'DownpourUnitAccessor', 'DownpourDoubleUnitAccessor', 'DownpourCtrDymfAccessor'], but actual %s"
                    % (accessor_class)
                )

            if accessor_class.find("Double") >= 0:
                table_data.accessor.accessor_class = 'CtrDoubleAccessor'
            elif accessor_class.find("Dymf") >= 0:
                table_data.accessor.accessor_class = 'CtrDymfAccessor'
            else:
                table_data.accessor.accessor_class = 'CtrCommonAccessor'

            if not configs.get("use_cvm", True):
                table_data.accessor.accessor_class = 'SparseAccessor'

            table_data.accessor.embedx_dim = config.get('sparse_embedx_dim', 8)
            table_data.accessor.fea_dim = table_data.accessor.embedx_dim + 3
            table_data.accessor.embedx_threshold = config.get(
                'sparse_embedx_threshold', 10
            )

            if accessor_class == 'DownpourUnitAccessor':
                table_data.accessor.ctr_accessor_param.show_scale = False
            else:
                table_data.accessor.ctr_accessor_param.show_scale = True

            table_data.accessor.ctr_accessor_param.nonclk_coeff = config.get(
                'sparse_nonclk_coeff', 0.1
            )
            table_data.accessor.ctr_accessor_param.click_coeff = config.get(
                'sparse_click_coeff', 1
            )
            table_data.accessor.ctr_accessor_param.base_threshold = config.get(
                'sparse_base_threshold', 1.5
            )
            table_data.accessor.ctr_accessor_param.delta_threshold = config.get(
                'sparse_delta_threshold', 0.25
            )
            table_data.accessor.ctr_accessor_param.delta_keep_days = config.get(
                'sparse_delta_keep_days', 16
            )
            table_data.accessor.ctr_accessor_param.show_click_decay_rate = (
                config.get('sparse_show_click_decay_rate', 0.98)
            )
            table_data.accessor.ctr_accessor_param.delete_threshold = (
                config.get('sparse_delete_threshold', 0.8)
            )
            table_data.accessor.ctr_accessor_param.delete_after_unseen_days = (
                config.get('sparse_delete_after_unseen_days', 30)
            )
            table_data.accessor.ctr_accessor_param.ssd_unseenday_threshold = (
                config.get('sparse_ssd_unseenday_threshold', 1)
            )
            load_filter_slots = config.get('sparse_load_filter_slots', [])
            table_data.accessor.ctr_accessor_param.load_filter_slots.extend(
                load_filter_slots
            )
            save_filter_slots = config.get('sparse_save_filter_slots', [])
            table_data.accessor.ctr_accessor_param.save_filter_slots.extend(
                save_filter_slots
            )
            converter = config.get('sparse_converter', "")
            deconverter = config.get('sparse_deconverter', "")

            save_data1 = table_data.accessor.table_accessor_save_param.add()
            save_data1.param = 1
            save_data1.converter = converter
            save_data1.deconverter = deconverter

            save_data2 = table_data.accessor.table_accessor_save_param.add()
            save_data2.param = 2
            save_data2.converter = converter
            save_data2.deconverter = deconverter

            if (
                accessor_class == 'DownpourCtrAccessor'
                or accessor_class == 'DownpourCtrDoubleAccessor'
            ):
                sparse_optimizer_config(
                    table_data.accessor.embed_sgd_param, config, ''
                )
                sparse_optimizer_config(
                    table_data.accessor.embedx_sgd_param, config, ''
                )
            else:
                sparse_optimizer_config(
                    table_data.accessor.embed_sgd_param, config, 'embed_'
                )
                sparse_optimizer_config(
                    table_data.accessor.embedx_sgd_param, config, 'embedx_'
                )
            add_graph_config(table_data.accessor.graph_sgd_param, config)

        if not configs:
            logger.info("fleet desc config is empty")
        else:
            for table_name in configs:
                if (
                    table_name == 'dense_table'
                    or table_name == 'datanorm_table'
                ):
                    continue
                if type(configs[table_name]) != dict:
                    continue
                table_data = table_param.add()
                table_data.table_name = table_name
                set_sparse_table_config(table_data, configs[table_name])

    @property
    def amp(self):
        """
        Indicating whether we are using automatic mixed precision training
        Default Value: False

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.amp = True # by default this is false

        """
        return self.strategy.amp

    @amp.setter
    @is_strict_auto
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            logger.warning("amp should have value of bool type")

    @property
    def amp_configs(self):
        """

        Set automatic mixed precision training configurations. In general, amp has several configurable
        settings that can be configured through a dict.

        **Notes**:
            init_loss_scaling(float): The initial loss scaling factor. Default 32768.

            use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling. Default True.

            incr_every_n_steps(int): Increases loss scaling every n consecutive steps with finite gradients. Default 1000.

            decr_every_n_nan_or_inf(int): Decreases loss scaling every n accumulated steps with nan or inf gradients. Default 2.

            incr_ratio(float): The multiplier to use when increasing the loss scaling. Default 2.0.

            decr_ratio(float): The less-than-one-multiplier to use when decreasing the loss scaling. Default 0.5.

            custom_white_list(list[str]): Users' custom white list which always execution fp16.

            custom_black_list(list[str]): Users' custom black list which forbidden execution fp16.

            custom_black_varnames(list[str]): Users' custom black variables' names.

            use_pure_fp16(bool): Whether to use the pure fp16 training. Default False.

            use_pure_bf16(bool): Whether to use the pure bf16 training. Default False.

            use_fp16_guard(bool): Whether to use `fp16_guard` when constructing the program.
            Default True. Only takes effect when `use_pure_fp16` is turned on.

        Examples:
            .. code-block:: python
                :name: example_1

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.amp = True
                >>> strategy.amp_configs = {
                ...     "init_loss_scaling": 32768,
                ...     "custom_white_list": ['conv2d']
                ... }

            .. code-block:: python
                :name: example_2

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.amp = True
                >>> # pure fp16
                >>> strategy.amp_configs = {
                ...     "init_loss_scaling": 32768,
                ...     "use_pure_fp16": True
                ... }

        """
        return get_msg_dict(self.strategy.amp_configs)

    @amp_configs.setter
    @is_strict_auto
    def amp_configs(self, configs):
        check_configs_key(self.strategy.amp_configs, configs, "amp_configs")
        assign_configs_value(self.strategy.amp_configs, configs)

    @property
    def asp(self):
        """

        Indicating whether we are using automatic sparsity training
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.asp = True # by default this is false

        """
        return self.strategy.asp

    @asp.setter
    @is_strict_auto
    def asp(self, flag):
        if isinstance(flag, bool):
            self.strategy.asp = flag
        else:
            logger.warning("asp should have value of bool type")

    @property
    def qat(self):
        """
        Indicating whether we are using quantization aware training
        Default Value: False

        Examples:

            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.qat = True # by default this is false

        """
        return self.strategy.qat

    @qat.setter
    @is_strict_auto
    def qat(self, flag):
        assert isinstance(flag, bool), "qat should have value of bool type"
        self.strategy.qat = flag

    @property
    def qat_configs(self):
        """
        Set quantization training configurations. In general, qat has several configurable
        settings that can be configured through a dict.
        **Notes**:
            channel_wise_abs_max(bool): Whether to use `per_channel` quantization training. Default is True.
            weight_bits(int): quantization bit number for weight. Default is 8.
            activation_bits(int): quantization bit number for activation. Default is 8.
            not_quant_pattern(list[str]): When the skip pattern is detected in an op's name scope,
                the corresponding op will not be quantized.
            algo(str): Other quantization training algorithm.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.qat = True
                >>> strategy.qat_configs = {
                ...     "channel_wise_abs_max": True,
                ...     "weight_bits": 8,
                ...     "activation_bits": 8,
                ...     "not_quant_pattern": ['skip_quant']
                ... }

        """
        return get_msg_dict(self.strategy.qat_configs)

    @qat_configs.setter
    def qat_configs(self, configs):
        check_configs_key(self.strategy.qat_configs, configs, "qat_configs")
        assign_configs_value(self.strategy.qat_configs, configs)

    @property
    def recompute(self):
        """
        Indicating whether we are using forward recomputation for memory optimization
        Default value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.recompute = True
                >>> # suppose x and y are names of checkpoint tensors for recomputation
                >>> strategy.recompute_configs = {"checkpoints": ["x", "y"]}

        """
        return self.strategy.recompute

    @property
    def sync_nccl_allreduce(self):
        """

        Indicating whether we are using synchronized all reduce in each communication thread
        We note that system overhead is usually lower when sync_nccl_allreduce = True

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.sync_nccl_allreduce = True

        """
        return self.strategy.sync_nccl_allreduce

    @sync_nccl_allreduce.setter
    @is_strict_auto
    def sync_nccl_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_nccl_allreduce = flag
        else:
            logger.warning("sync_nccl_allreduce should have value of bool type")

    @property
    def use_hierarchical_allreduce(self):
        """

        Indicating whether we are using hierarchical allreduce in collective communication
        Hierarchical allreduce often does allreduce within a certain node group and then do
        allreduce among the leaders of each group

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.use_hierarchical_allreduce = True

        """
        return self.strategy.use_hierarchical_allreduce

    @use_hierarchical_allreduce.setter
    @is_strict_auto
    def use_hierarchical_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.use_hierarchical_allreduce = flag
        else:
            logger.warning(
                "use_hierarchical_allreduce should have value of bool type"
            )

    @property
    def hierarchical_allreduce_inter_nranks(self):
        """

        Number of ranks for low level node groups in hierarchical allreduce
        Default value: number of GPU cards on each single GPU machine

        Example:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.hierarchical_allreduce_inter_nranks = 8

        """
        return self.strategy.hierarchical_allreduce_inter_nranks

    @hierarchical_allreduce_inter_nranks.setter
    @is_strict_auto
    def hierarchical_allreduce_inter_nranks(self, value):
        if isinstance(value, int):
            self.strategy.hierarchical_allreduce_inter_nranks = value
        else:
            logger.warning(
                "hierarchical_allreduce_inter_nranks should have value of int type"
            )

    @property
    def sync_batch_norm(self):
        """

        Indicating whether we are using sync_batch_norm to do synchronous batch normalization among all training nodes.

        Default value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.sync_batch_norm = True

        """

        return self.strategy.sync_batch_norm

    @sync_batch_norm.setter
    @is_strict_auto
    def sync_batch_norm(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_batch_norm = flag
        else:
            logger.warning("sync_batch_norm should have value of bool type")

    @property
    def fuse_all_reduce_ops(self):
        """

        Indicating whether we are using fuse_all_reduce_ops for gradient fusion during backward phase of training
        Default value: True

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.fuse_all_reduce_ops = False

        """
        return self.strategy.fuse_all_reduce_ops

    @fuse_all_reduce_ops.setter
    @is_strict_auto
    def fuse_all_reduce_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_all_reduce_ops = flag
        else:
            logger.warning("fuse_all_reduce_ops should have value of bool type")

    @property
    def fuse_grad_size_in_MB(self):
        """

        Specifying the size of gradient to fuse in Mega-Bytes

        Default value: 32

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.fuse_grad_size_in_MB = 50

        """
        return self.strategy.fuse_grad_size_in_MB

    @fuse_grad_size_in_MB.setter
    @is_strict_auto
    def fuse_grad_size_in_MB(self, value):
        if isinstance(value, int):
            self.strategy.fuse_grad_size_in_MB = value
        else:
            logger.warning("fuse_grad_size_in_MB should have value of int type")

    @property
    def last_comm_group_size_MB(self):
        """

        Specifying the size of gradient to fuse in Mega-Bytes when
        the last group of each batch communicates. Making the last group
        small is useful to improve performance.

        Default value: 1

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.last_comm_group_size_MB = 2

        """
        return self.strategy.last_comm_group_size_MB

    @last_comm_group_size_MB.setter
    @is_strict_auto
    def last_comm_group_size_MB(self, value):
        if value > 0:
            self.strategy.last_comm_group_size_MB = value
        else:
            raise ValueError("last_comm_group_size_MB should be greater than 0")

    @property
    def find_unused_parameters(self):
        """

        Indicating whether we are using find_unused_parameters to
        find unused parameters in DataParallel.

        Default value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.find_unused_parameters = True

        """

        return self.strategy.find_unused_parameters

    @find_unused_parameters.setter
    @is_strict_auto
    def find_unused_parameters(self, flag):
        if isinstance(flag, bool):
            self.strategy.find_unused_parameters = flag
        else:
            logger.warning(
                "find_unused_parameters should have value of bool type"
            )

    @property
    def _fuse_grad_size_in_TFLOPS(self):
        return self.strategy.fuse_grad_size_in_TFLOPS

    @_fuse_grad_size_in_TFLOPS.setter
    @is_strict_auto
    def _fuse_grad_size_in_TFLOPS(self, value):
        if isinstance(value, float):
            self.strategy.fuse_grad_size_in_TFLOPS = value
        else:
            logger.warning(
                "fuse_grad_size_in_TFLOPS should have value of float type"
            )

    @property
    def nccl_comm_num(self):
        """

        Specifying the number of NCCL communicator

        Default value: 1

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.nccl_comm_num = 2

        """

        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    @is_strict_auto
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            logger.warning("nccl_comm_num should have value of int type")

    @recompute.setter
    @is_strict_auto
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            logger.warning("recompute should have value of bool type")

    @property
    def recompute_configs(self):
        """

        Set recompute configurations.

        **Note**:
        checkpoints(list): list of string name of checkpoints. In general, the recompute
        strategy of current implementation should have some manually assign checkpoints.

        enable_offload(bool): enable recompute checkpoints offload feature. this feature
        will offload the checkpoint to host memory to allow even larger batch size. since
        the memcpy from host to device takes time, it is a trade off between larger batch
        size and training speed.

        checkpoint_shape(list): list of int that specific the shape of checkpoint. so far
        recompute-offload requires that all checkpoint to be same shape, and every dimension
        specific here should be determined ("-1" is not allowed).

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.recompute = True
                >>> strategy.recompute_configs = {
                ...     "checkpoints": ["x", "y"],
                ...     "enable_offload": True,
                ...     "checkpoint_shape": [100, 512, 1024]
                ... }

        """
        return get_msg_dict(self.strategy.recompute_configs)

    @recompute_configs.setter
    @is_strict_auto
    def recompute_configs(self, configs):
        check_configs_key(
            self.strategy.recompute_configs, configs, "checkpoint_configs"
        )
        assign_configs_value(self.strategy.recompute_configs, configs)

    @property
    def sharding(self):
        """

        Indicating whether we are using sharding Optimizer for memory
        optimization. We implement the sharding optimizer following the ZeRO-DP
        idea from [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054).
        Model parameters and Optimizer State are sharded into different ranks allowing to fit larger model.

        In Hybrid parallelism scenario, we use sharding config as uniform API to set each parallelism.

        Default value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.sharding = True

        """
        return self.strategy.sharding

    @sharding.setter
    @is_strict_auto
    def sharding(self, flag):
        if isinstance(flag, bool):
            self.strategy.sharding = flag
        else:
            logger.warning("sharding should have value of bool type")

    @property
    def sharding_configs(self):
        """

        Set sharding configurations.

        **Note**:
            sharding_segment_strategy(string, optional): strategy used to segment the program(forward & backward operations). two strategise are
            available: "segment_broadcast_MB" and "segment_anchors". segment is a concept used in sharding to overlap computation and
            communication. Default is segment_broadcast_MB.

            segment_broadcast_MB(float, optional): segment by the parameters broadcast volume. sharding will introduce parameter broadcast operations into program, and
            after every segment_broadcast_MB size parameter being broadcasted, the program will be cut into one segment.
            This configuration will affect the communication speed in sharding training, and should be an empirical value decided by your model size and network topology.
            Only enable when sharding_segment_strategy = segment_broadcast_MB. Default is 32.0 .

            segment_anchors(list): list of anchors used to segment the program, which allows a finer control of program segmentation.
            this strategy is experimental by now. Only enable when sharding_segment_strategy = segment_anchors.

            sharding_degree(int, optional): specific the number of gpus within each sharding parallelism group; and sharding will be turn off if sharding_degree=1.  Default is 8.

            gradient_merge_acc_step(int, optional): specific the accumulation steps in gradient merge; and gradient merge will be turn off if gradient_merge_acc_step=1.  Default is 1.

            optimize_offload(bool, optional): enable the optimizer offload which will offload the moment vars to Host memory in order to saving GPU memory for fitting larger model.
            the moment var will be prefetch from and offloaded to Host memory during update stage. it is a strategy that trades off between training speed and GPU memory, and is recommended to be turn on only when gradient_merge_acc_step large, where
            the number of time of update stage will be relatively small compared with forward&backward's.  Default is False.

            dp_degree(int, optional): specific the number of data parallelism group; when dp_degree >= 2, it will introduce dp_degree ways data parallelism as the outer parallelism for the inner parallelism. User is responsible to ensure global_world_size = mp_degree * sharding_degree * pp_degree * dp_degree. Default is 1.

            mp_degree(int, optional): [Hybrid parallelism ONLY] specific the number of gpus within each megatron parallelism group; and megatron parallelism will turn be off if mp_degree=1.  Default is 1.

            pp_degree(int, optional): [Hybrid parallelism ONLY] specific the number of gpus within each pipeline parallelism group; and pipeline parallelism will turn be off if pp_degree=1.  Default is 1.

            pp_allreduce_in_optimize(bool, optional): [Hybrid parallelism ONLY] move the allreduce operations from backward stage to update(optimize) stage when pipeline parallelism is on.
            This configuration will affect the communication speed of Hybrid parallelism training depended on network topology. this strategy is experimental by now..  Default is False.

            optimize_cast(bool, optional): [Hybrid parallelism ONLY] Move the cast op of AMP which cast fp32 param to fp16 param to optimizer. optimize_cast will persist fp16 param, it
            will take more memory, but will be faster, trade space for time. Recommend to turn on only when using pipeline or gradient_merge_acc_step large.


        Examples:
            .. code-block:: python

                >>> # sharding-DP, 2 nodes with 8 gpus per node
                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.sharding = True
                >>> strategy.sharding_configs = {
                ...     "sharding_segment_strategy": "segment_broadcast_MB",
                ...     "segment_broadcast_MB": 32,
                ...     "sharding_degree": 8,
                ...     "dp_degree": 2,
                ...     "gradient_merge_acc_step": 4,
                ... }

        """
        return get_msg_dict(self.strategy.sharding_configs)

    @sharding_configs.setter
    @is_strict_auto
    def sharding_configs(self, configs):
        check_configs_key(
            self.strategy.sharding_configs, configs, "sharding_configs"
        )
        assign_configs_value(self.strategy.sharding_configs, configs)

    @property
    def without_graph_optimization(self):
        """

        Run program using Executor other than ParallelExecutor.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.without_graph_optimization = True

        """
        return self.strategy.without_graph_optimization

    @without_graph_optimization.setter
    @is_strict_auto
    def without_graph_optimization(self, flag):
        if isinstance(flag, bool):
            self.strategy.without_graph_optimization = flag
        else:
            logger.warning(
                "without_graph_optimization should have value of bool type"
            )

    @property
    def _calc_comm_same_stream(self):
        """

        This based on raw_program_optimizer program
        Set whether use same stream for calc and comm when fuse allreduce
        The default value for the calc_comm_same_stream is False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy._calc_comm_same_stream = True

        """
        return self.strategy.calc_comm_same_stream

    @_calc_comm_same_stream.setter
    @is_strict_auto
    def _calc_comm_same_stream(self, same):
        if isinstance(same, bool):
            self.strategy.calc_comm_same_stream = same
        else:
            logger.warning(
                "calc_comm_same_stream should have value of boolean type"
            )

    @property
    def fuse_grad_merge(self):
        """

        Set whether fuse the grad for gradient merge.
        Note: this flag will only effect the gradient merge under pipeline mode
        The default value for the fuse_grad_merge is False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.fuse_grad_merge = True

        """
        return self.strategy.fuse_grad_merge

    @fuse_grad_merge.setter
    @is_strict_auto
    def fuse_grad_merge(self, fuse_grad_merge):
        if isinstance(fuse_grad_merge, bool):
            self.strategy.fuse_grad_merge = fuse_grad_merge
        else:
            logger.warning("fuse_grad_merge should have value of boolean type")

    @property
    def fuse_grad_size_in_num(self):
        """

        This based on raw_program_optimizer program and allreduce the num of the fused op

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.fuse_grad_size_in_num = 2

        """
        return self.strategy.fuse_grad_size_in_num

    @fuse_grad_size_in_num.setter
    @is_strict_auto
    def fuse_grad_size_in_num(self, num):
        if isinstance(num, int):
            self.strategy.fuse_grad_size_in_num = num
        else:
            logger.warning(
                "fuse_grad_size_in_num should have value of int32 type"
            )

    @property
    def pipeline(self):
        """

        Indicating whether we are using pipeline parallelism for distributed training.
        Current implementation mainly focus on single GPU machine pipeline parallelism and
        data parallelism across GPU machine. The pipeline information is indicated through
        device_guard information in user-defined program.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.pipeline = True

        """
        return self.strategy.pipeline

    @property
    def is_fl_ps_mode(self):
        return self.strategy.is_fl_ps_mode

    @is_fl_ps_mode.setter
    @is_strict_auto
    def is_fl_ps_mode(self, flag):
        if isinstance(flag, bool):
            self.strategy.is_fl_ps_mode = flag
        else:
            logger.warning("is_fl_ps_mode should have value of bool type")

    @property
    def is_with_coordinator(self):
        return self.strategy.with_coordinator

    @is_with_coordinator.setter
    @is_strict_auto
    def is_with_coordinator(self, flag):
        if isinstance(flag, bool):
            self.strategy.with_coordinator = flag
        else:
            logger.warning("with_coordinator should have value of bool type")

    @pipeline.setter
    @is_strict_auto
    def pipeline(self, flag):
        if isinstance(flag, bool):
            self.strategy.pipeline = flag
        else:
            logger.warning("pipeline should have value of bool type")

    @property
    def pipeline_configs(self):
        """

        Set pipeline parallelism configurations. In pipeline parallelism,
        different parts of neural networks are running on different GPUS.
        There are Tensor queue buffer between each pair of neighborhood GPUS
        that are responsible for synchronizing hidden Tensor results between
        GPUs. Pipeline parallelism consists of several producer-consumer style
        hardware pairs, such as GPU-GPU, CPU-GPU, GPU-XPU. The best way to speedup
        pipeline parallelism is to make the size of Tensor in Tensor queue smaller,
        so that we will have a faster producer for downstream consumers.

        **Notes**:
            **Detailed arguments for pipeline_configs**

            **micro_batch_size**: the number of small batches in each user defined batch

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.pipeline = True
                >>> strategy.pipeline_configs = {"micro_batch_size": 12}

        """

        return get_msg_dict(self.strategy.pipeline_configs)

    @pipeline_configs.setter
    @is_strict_auto
    def pipeline_configs(self, configs):
        check_configs_key(
            self.strategy.pipeline_configs, configs, "pipeline_configs"
        )
        assign_configs_value(self.strategy.pipeline_configs, configs)

    @property
    def tensor_parallel(self):
        """

        Indicating whether we are using tensor parallel for distributed training.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.tensor_parallel = True

        """
        return self.strategy.tensor_parallel

    @tensor_parallel.setter
    @is_strict_auto
    def tensor_parallel(self, flag):
        if isinstance(flag, bool):
            self.strategy.tensor_parallel = flag
        else:
            logger.warning("tensor_parallel should have value of bool type")

    @property
    def tensor_parallel_configs(self):
        """

        Set tensor_parallel configurations.

        **Notes**:
            **Detailed arguments for tensor_parallel_configs**

            **tensor_parallel_degree**: degree of tensor parallel

            **tensor_init_seed**: parameter initialization random seed


        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.tensor_parallel = True
                >>> strategy.tensor_parallel_configs = {"tensor_parallel_degree": 4,
                ...                                     "tensor_init_seed": 123}

        """
        return get_msg_dict(self.strategy.tensor_parallel_configs)

    @tensor_parallel_configs.setter
    @is_strict_auto
    def tensor_parallel_configs(self, configs):
        check_configs_key(
            self.strategy.tensor_parallel_configs,
            configs,
            "tensor_parallel_configs",
        )
        assign_configs_value(self.strategy.tensor_parallel_configs, configs)

    @property
    def hybrid_configs(self):
        """

        Dynamic graph hybrid parallel strategy configuration. Five-way hybrid parallelism
        needs to meet the following relationships

        total_number_GPUs = dp_degree * mp_degree * pp_degree * sharding_degree * sep_degree

        **Note**:
            **dp_degree(int)**: set number of GPUs in a data parallel group. Default -1.
                                    This value should be an integer greater than 0.
                                    If it is not set, or set to -1, its value will be inferred
                                    based on the total number of cards.

            **mp_degree(int)**: set number of GPUs in a model parallel group. Default 1

            **pp_degree(int)**: set number of GPUs in a pipeline parallel group. Default 1
            **sep_degree(int)**: set number of GPUs in a sep parallel group. Default 1
            **sharding_degree(int)**: set number of GPUs in a sharding parallel group. Default 1
            **order(list(string))**: set hybrid parallel dimensions, the order is from outside to inside. Default ['dp','pp','sharding','sep', 'mp']

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.hybrid_configs = {
                ...     "dp_degree": 1,
                ...     "mp_degree": 2,
                ...     "pp_degree": 1,
                ...     "order":['dp','pp','sharding', 'sep', 'mp']
                ... }

        """
        return get_msg_dict(self.strategy.hybrid_configs)

    @hybrid_configs.setter
    def hybrid_configs(self, configs):
        hybrid_config = copy.deepcopy(configs)
        if "order" in hybrid_config:
            self.hybrid_parallel_order = hybrid_config["order"]
            hybrid_config.pop('order')

        check_configs_key(
            self.strategy.hybrid_configs, hybrid_config, "hybrid_configs"
        )

        if "mp_configs" in configs:
            if "sync_param_name" in configs["mp_configs"]:
                self.sync_param_name = configs["mp_configs"]["sync_param_name"]
                configs["mp_configs"].pop("sync_param_name")

            assign_configs_value(
                self.strategy.hybrid_configs.mp_configs, configs["mp_configs"]
            )
            configs.pop("mp_configs")
        if "pp_configs" in configs:
            assign_configs_value(
                self.strategy.hybrid_configs.pp_configs, configs["pp_configs"]
            )
            configs.pop("pp_configs")

        assign_configs_value(self.strategy.hybrid_configs, configs)

    @property
    def localsgd(self):
        """

        Indicating whether we are using Local SGD training. Default Value: False
        For more details, please refer to
        `Don't Use Large Mini-Batches, Use Local SGD <https://arxiv.org/pdf/1808.07217.pdf>`_.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.localsgd = True # by default this is false

        """
        return self.strategy.localsgd

    @localsgd.setter
    @is_strict_auto
    def localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.localsgd = flag
        else:
            logger.warning("localsgd should have value of bool type")

    @property
    def localsgd_configs(self):
        """

        Set LocalSGD training configurations. LocalSGD has a configurable
        setting that can be configured through a dict.

        **Notes**:
            k_steps(int) The local steps for training before parameter synchronization. Default 1.
            begin_step(int) The step of beginning training by localsgd. Default 1.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.localsgd = True
                >>> strategy.localsgd_configs = {"k_steps": 4,
                ...                             "begin_step": 30}

        """

        return get_msg_dict(self.strategy.localsgd_configs)

    @localsgd_configs.setter
    @is_strict_auto
    def localsgd_configs(self, configs):
        check_configs_key(
            self.strategy.localsgd_configs, configs, "localsgd_configs"
        )
        assign_configs_value(self.strategy.localsgd_configs, configs)

    @property
    def adaptive_localsgd(self):
        """

        Indicating whether we are using Adaptive Local SGD training. Default Value: False
        For more details, please refer to `Adaptive Communication Strategies to Achieve
        the Best Error-Runtime Trade-off in Local-Update SGD <https://arxiv.org/pdf/1810.08313.pdf>`_.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.adaptive_localsgd = True # by default this is false

        """
        return self.strategy.adaptive_localsgd

    @adaptive_localsgd.setter
    @is_strict_auto
    def adaptive_localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.adaptive_localsgd = flag
        else:
            logger.warning("adaptive_localsgd should have value of bool type")

    @property
    def adaptive_localsgd_configs(self):
        """

        Set AdaptiveLocalSGD training configurations. AdaptiveLocalSGD has a configurable
        setting that can be configured through a dict.

        **Notes**:
            init_k_steps(int) The initial steps for training before adaptive localsgd.
                              Then, the adaptive localsgd method will modify init_k_steps automatically.
                              Default 1.

            begin_step(int) The step of beginning training by adaptive localsgd. Default 1.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.adaptive_localsgd = True
                >>> strategy.adaptive_localsgd_configs = {"init_k_steps": 1,
                ...                                       "begin_step": 30}

        """

        return get_msg_dict(self.strategy.adaptive_localsgd_configs)

    @adaptive_localsgd_configs.setter
    @is_strict_auto
    def adaptive_localsgd_configs(self, configs):
        check_configs_key(
            self.strategy.adaptive_localsgd_configs,
            configs,
            "adaptive_localsgd_configs",
        )
        assign_configs_value(self.strategy.adaptive_localsgd_configs, configs)

    @property
    def dgc(self):
        """

        Indicating whether we are using Deep Gradient Compression training. For more details, please refer to
        [Deep Gradient Compression](https://arxiv.org/abs/1712.01887).

        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.dgc = True # by default this is false

        """
        return self.strategy.dgc

    @dgc.setter
    @is_strict_auto
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            logger.warning("dgc should have value of bool type")

    @property
    def dgc_configs(self):
        r"""

        Set Deep Gradient Compression training configurations. In general, dgc has several configurable
        settings that can be configured through a dict.

        **Notes**:
            rampup_begin_step(int): The beginning step from which gradient compression is implemented. Default 0.

            rampup_step(int): Time steps used in sparsity warm-up periods. Default is 1. \
                    For example, if the sparsity is [0.75, 0.9375, 0.984375, 0.996, 0.999], and the rampup_step is 100, \
                    it will use 0.75 at 0~19 steps, and 0.9375 at 20~39 steps, and so on. And when reach sparsity array \
                    ends, it will use 0.999 then and after.

            sparsity(list[float]): Get top important element from gradient tensor, the ratio is (1 - sparsity). \
                    Default is [0.999]. For example, if the sparsity is [0.99, 0.999], the top [1%, 0.1%] important \
                    element will be transmitted.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.dgc = True
                >>> strategy.dgc_configs = {"rampup_begin_step": 1252}

        """
        return get_msg_dict(self.strategy.dgc_configs)

    @dgc_configs.setter
    @is_strict_auto
    def dgc_configs(self, configs):
        check_configs_key(self.strategy.dgc_configs, configs, "dgc_configs")
        assign_configs_value(self.strategy.dgc_configs, configs)

    @property
    def fp16_allreduce(self):
        """

        Indicating whether we are using fp16 gradient allreduce training
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.fp16_allreduce = True # by default this is false

        """
        return self.strategy.fp16_allreduce

    @fp16_allreduce.setter
    @is_strict_auto
    def fp16_allreduce(self, flag):
        if not isinstance(flag, bool):
            raise TypeError('fp16_allreduce must be value of bool type')
        self.strategy.fp16_allreduce = flag

    @property
    def gradient_merge(self):
        """

        Gradient Merge, also called as Gradient Accumulation,
        is a strategy for large batch training. With this strategy,
        model parameter will not be updated until user-defined steps.
        For each step, the forward network and the backward network
        will run to calculate the gradient of model parameters.
        For every k step, the optimization network will run,
        applying a specific optimization method (such as SGD, Adam)
        to model parameters.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.gradient_merge = True
                >>> strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}

        """
        return self.strategy.gradient_merge

    @gradient_merge.setter
    @is_strict_auto
    def gradient_merge(self, flag):
        if isinstance(flag, bool):
            self.strategy.gradient_merge = flag
        else:
            logger.warning("gradient_merge should have value of bool type")

    @property
    def gradient_merge_configs(self):
        """

        the key-value configs of distribute_strategy

        **Note**:
            k_steps(int): the update period of the parameters.

            avg(bool): whether to average the gradients of each mini-batch, the default value is `True`

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.gradient_merge = True
                >>> strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}

        """
        return get_msg_dict(self.strategy.gradient_merge_configs)

    @gradient_merge_configs.setter
    @is_strict_auto
    def gradient_merge_configs(self, configs):
        check_configs_key(
            self.strategy.gradient_merge_configs, configs, "gradient_configs"
        )
        assign_configs_value(self.strategy.gradient_merge_configs, configs)

    @property
    def lars(self):
        """

        Set lars configurations. lars is used to deal with the convergence problems when the global
        batch size is larger than 8k.  For more details, please refer to
        [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888).

        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.lars = True # by default this is false

        """
        return self.strategy.lars

    @lars.setter
    @is_strict_auto
    def lars(self, flag):
        if isinstance(flag, bool):
            self.strategy.lars = flag
        else:
            logger.warning("lars should have value of bool type")

    @property
    def lars_configs(self):
        """

        Set Lars training configurations.

        **Notes**:
        **lars_coeff (float)**: trust ratio in lars formula.
        **lars_weight_decay** (float): weight decay coefficient in lars formula.
        **epsilon (float)**: argument is used to avoid potential division-by-zero
        when compute the local lr;
        **exclude_from_weight_decay ([string])**: is a list of name strings of layers which
        will be exclude from weight decay in lars formula.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.lars = True
                >>> strategy.lars_configs = {
                ...             "lars_coeff": 0.01,
                ...             "lars_weight_decay": 0.0005,
                ...             "epsilon": 0,
                ...             "exclude_from_weight_decay": ['batch_norm', '.b_0']
                ... }

        """
        return get_msg_dict(self.strategy.lars_configs)

    @lars_configs.setter
    @is_strict_auto
    def lars_configs(self, configs):
        check_configs_key(self.strategy.lars_configs, configs, "lars_configs")
        assign_configs_value(self.strategy.lars_configs, configs)

    @property
    def lamb(self):
        """

        Set lamb configurations. lamb is used to deal with the convergence problems for large
        batch size training, specially for attention-related model like BERT. For more details,
        please refer to
        [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962).

        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.lamb = True # by default this is false

        """

        return self.strategy.lamb

    @lamb.setter
    @is_strict_auto
    def lamb(self, flag):
        if isinstance(flag, bool):
            self.strategy.lamb = flag
        else:
            logger.warning("lamb should have value of bool type")

    @property
    def lamb_configs(self):
        """

        Set Lars training configurations.

        **Notes**:
        **lamb_weight_decay** (float): weight decay coefficient in lamb formula.
        **exclude_from_weight_decay ([string])**: is a list of name strings of layers which
        will be exclude from weight decay in lamb formula.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.lamb = True
                >>> strategy.lamb_configs = {
                ...         'lamb_weight_decay': 0.01,
                ...         'exclude_from_weight_decay': [],
                ... }

        """
        return get_msg_dict(self.strategy.lamb_configs)

    @lamb_configs.setter
    @is_strict_auto
    def lamb_configs(self, configs):
        check_configs_key(self.strategy.lamb_configs, configs, "lamb_configs")
        assign_configs_value(self.strategy.lamb_configs, configs)

    @property
    def elastic(self):
        """

        Indicating whether we want to do current distributed training on clusters with elastic resources.
        Currently, this is configuration is not valid.

        """
        return self.strategy.elastic

    @elastic.setter
    @is_strict_auto
    def elastic(self, flag):
        if isinstance(flag, bool):
            self.strategy.elastic = flag
        else:
            logger.warning("elastic should have value of bool type")

    @property
    def auto(self):
        """

        Indicating whether we are using auto-parallel configuration
        This feature is currently an experimental feature. Currently,
        auto-parallelism can be used only when a user does not set any other
        strategy configs except auto. For details, please reference the following
        code example
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.auto = True
                >>> # if set other strategy at the same time, auto will not apply
                >>> # strategy.amp = True

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.auto

    @auto.setter
    def auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto = flag
        else:
            logger.warning("auto should have value of bool type")

    @property
    def semi_auto(self):
        """

        Indicating whether we are using semi-auto parallel function
        This feature is currently an experimental feature. Currently,
        auto-parallelism can be used only when a user does not set any other
        strategy configs except semi-auto. For details, please reference the following
        code example
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.semi_auto = True
                >>> # if set other strategy at the same time, auto will not apply
                >>> # strategy.amp = True

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.semi_auto

    @semi_auto.setter
    def semi_auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.semi_auto = flag
        else:
            logger.warning("semi-auto should have value of bool type")

    @property
    def auto_search(self):
        """

        Indicating whether we are using auto-search parallel function
        For details, please reference the following code example
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.auto_search = True

        """
        return self.strategy.auto_search

    @auto_search.setter
    def auto_search(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto_search = flag
        else:
            logger.warning("auto-search should have value of bool type")

    @property
    def split_data(self):
        """

        Indicating whether we split the data. If True, we split the data.
        Default Value: True

        Examples:
            .. code-block:: python

                >>> import paddle

                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet
                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.split_data = True

        """
        return self.strategy.split_data

    @split_data.setter
    def split_data(self, flag):
        if isinstance(flag, bool):
            self.strategy.split_data = flag
        else:
            logger.warning("split_data should have value of bool type")

    @property
    def qat(self):
        """

        Indicating whether we are using quantization training
        Default Value: False

        """
        return self.strategy.qat

    @qat.setter
    def qat(self, flag):
        if isinstance(flag, bool):
            self.strategy.qat = flag
        else:
            logger.warning("qat should have value of bool type")

    @property
    def qat_configs(self):
        """

        Set quantization training configurations. In general, qat has several configurable
        settings that can be configured through a dict.

        **Notes**:
            channel_wise_abs_max(bool): Whether to use `per_channel` quantization training. Default is True.

            weight_bits(int): quantization bit number for weight. Default is 8.

            activation_bits(int): quantization bit number for activation. Default is 8.

            not_quant_pattern(list[str]): When the skip pattern is detected in an op's name scope,
                the corresponding op will not be quantized.

            algo(str): Other quantization training algorithm.

        Examples:
            .. code-block:: python

                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.qat = True
                >>> strategy.qat_configs = {
                ...     "channel_wise_abs_max": True,
                ...     "weight_bits": 8,
                ...     "activation_bits": 8,
                ...     "not_quant_pattern": ['skip_quant']
                ... }

        """
        return get_msg_dict(self.strategy.qat_configs)

    @qat_configs.setter
    def qat_configs(self, configs):
        check_configs_key(self.strategy.qat_configs, configs, "qat_configs")
        assign_configs_value(self.strategy.qat_configs, configs)

    @property
    def heter_ccl_mode(self):
        """

        Indicating whether we are using heter_ccl_mode for model training.
        This feature is currently an experimental feature. Currently,
        heter_ccl_mode can be used only for dataparallel with dygraph mode.
        Default Value: False

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.heter_ccl_mode = True

                >>> # for initialize parallel env, only need to call
                >>> paddle.distributed.init_parallel_env()
                >>> # then the heterogenous context will be created.

        """
        return self.strategy.heter_ccl_mode

    @heter_ccl_mode.setter
    def heter_ccl_mode(self, flag):
        if isinstance(flag, bool):
            self.strategy.heter_ccl_mode = flag
        else:
            logger.warning("heter_ccl_mode should have value of bool type")

    @property
    def cudnn_exhaustive_search(self):
        """

        Indicating whether to use exhaustive search method to choose convolution algorithms.
        Exhaustive search attempts all cuDNN algorithms to choose the fastest algorithm.
        This method is time-consuming, the chosen algorithm will be cached for the given layer specifications.
        Once the layer specifications (like batch size, feature map size) are changed, it will search again.
        Default Value: True

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.cudnn_exhaustive_search = False

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.cudnn_exhaustive_search

    @cudnn_exhaustive_search.setter
    @is_strict_auto
    def cudnn_exhaustive_search(self, flag):
        if isinstance(flag, bool):
            self.strategy.cudnn_exhaustive_search = flag
        else:
            logger.warning(
                "cudnn_exhaustive_search should have value of bool type"
            )

    @property
    def conv_workspace_size_limit(self):
        """

        The workspace limit size in MB unit for choosing cuDNN convolution algorithms.
        The inner function of cuDNN obtain the fastest suited algorithm that fits within this memory limit.
        Usually, large workspace size may lead to choose faster algorithms,
        but significant increasing memory workspace. Users need to trade-off between memory and speed.
        Default Value: 4000

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.conv_workspace_size_limit = 1024

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.conv_workspace_size_limit

    @conv_workspace_size_limit.setter
    @is_strict_auto
    def conv_workspace_size_limit(self, value):
        if isinstance(value, int):
            self.strategy.conv_workspace_size_limit = value
        else:
            logger.warning(
                "conv_workspace_size_limit should have value of int type"
            )

    @property
    def cudnn_batchnorm_spatial_persistent(self):
        """

        Indicates whether to use the mode CUDNN_BATCHNORM_SPATIAL_PERSISTENT function in batchnorm.
        This is only useful in cudnn.
        Default Value: True

        Examples:
            .. code-block:: python

                >>> import paddle
                >>> paddle.enable_static()
                >>> import paddle.distributed.fleet as fleet

                >>> strategy = fleet.DistributedStrategy()
                >>> strategy.cudnn_batchnorm_spatial_persistent = True

                >>> optimizer = paddle.optimizer.SGD(learning_rate=0.01)
                >>> optimizer = fleet.distributed_optimizer(optimizer, strategy)

        """
        return self.strategy.cudnn_batchnorm_spatial_persistent

    @cudnn_batchnorm_spatial_persistent.setter
    @is_strict_auto
    def cudnn_batchnorm_spatial_persistent(self, flag):
        if isinstance(flag, bool):
            self.strategy.cudnn_batchnorm_spatial_persistent = flag
        else:
            logger.warning(
                "cudnn_batchnorm_spatial_persistent should have value of bool type"
            )

    def _enable_env(self):
        strategy = self.strategy
        keys = [
            "FLAGS_cudnn_batchnorm_spatial_persistent",
            "FLAGS_conv_workspace_size_limit",
            "FLAGS_cudnn_exhaustive_search",
            "FLAGS_sync_nccl_allreduce",
            "FLAGS_fuse_parameter_memory_size",
            "FLAGS_fuse_parameter_groups_size",
        ]
        values = [
            bool(strategy.cudnn_batchnorm_spatial_persistent),
            int(strategy.conv_workspace_size_limit),
            bool(strategy.cudnn_exhaustive_search),
            bool(strategy.sync_nccl_allreduce),
            int(strategy.fuse_grad_size_in_MB),
            int(strategy.fuse_grad_size_in_TFLOPS),
        ]

        for i, key in enumerate(keys):
            if _global_flags().is_public(key):
                _global_flags()[key] = values[i]

    def _is_strict_auto(self):
        global non_auto_func_called
        if self.strategy.auto and non_auto_func_called:
            return True
        return False

    def __repr__(self):
        spacing = 2
        max_k = 38
        max_v = 38

        length = max_k + max_v + spacing

        h1_format = "    " + f"|{{:^{length}s}}|\n"
        h2_format = "    " + "|{{:>{}s}}{}{{:^{}s}}|\n".format(
            max_k, " " * spacing, max_v
        )

        border = "    +" + "".join(["="] * length) + "+"
        line = "    +" + "".join(["-"] * length) + "+"

        draws = border + "\n"
        draws += h1_format.format("")
        draws += h1_format.format("DistributedStrategy Overview")
        draws += h1_format.format("")

        fields = self.strategy.DESCRIPTOR.fields
        str_res = ""

        env_draws = line + "\n"
        for f in fields:
            if "build_strategy" in f.name:
                continue
            if "_configs" in f.name:
                continue
            else:
                if isinstance(getattr(self.strategy, f.name), bool):
                    if hasattr(self.strategy, f.name + "_configs"):
                        if getattr(self.strategy, f.name):
                            draws += border + "\n"
                            draws += h1_format.format(
                                f"{f.name}=True <-> {f.name}_configs"
                            )
                            draws += line + "\n"
                            my_configs = getattr(
                                self.strategy, f.name + "_configs"
                            )
                            config_fields = my_configs.DESCRIPTOR.fields
                            protobuf_version = google.protobuf.__version__
                            if protobuf_version >= "4.21.0":
                                RepeatedScalarContainer = (
                                    google._upb._message.RepeatedScalarContainer
                                )
                            else:
                                from google.protobuf.pyext import _message

                                RepeatedScalarContainer = (
                                    _message.RepeatedScalarContainer
                                )
                            for ff in config_fields:
                                if isinstance(
                                    getattr(my_configs, ff.name),
                                    RepeatedScalarContainer,
                                ):
                                    values = getattr(my_configs, ff.name)
                                    for i, v in enumerate(values):
                                        if i == 0:
                                            draws += h2_format.format(
                                                ff.name, str(v)
                                            )
                                        else:
                                            draws += h2_format.format(
                                                "", str(v)
                                            )
                                else:
                                    draws += h2_format.format(
                                        ff.name,
                                        str(getattr(my_configs, ff.name)),
                                    )
                    else:
                        env_draws += h2_format.format(
                            f.name, str(getattr(self.strategy, f.name))
                        )
                else:
                    env_draws += h2_format.format(
                        f.name, str(getattr(self.strategy, f.name))
                    )

        result_res = (
            draws
            + border
            + "\n"
            + h1_format.format("Environment Flags, Communication Flags")
        )
        result_res += env_draws

        build_strategy_str = border + "\n"
        build_strategy_str += h1_format.format("Build Strategy")
        build_strategy_str += line + "\n"

        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            build_strategy_str += h2_format.format(
                f.name, str(getattr(self.strategy.build_strategy, f.name))
            )
        build_strategy_str += border + "\n"

        result_res += build_strategy_str
        return result_res

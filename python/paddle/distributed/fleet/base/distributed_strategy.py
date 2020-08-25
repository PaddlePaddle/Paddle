#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.distributed.fleet.proto import distributed_strategy_pb2
from paddle.fluid.framework import Variable
import google.protobuf.text_format


def get_msg_dict(msg):
    res_dict = {}
    fields = msg.DESCRIPTOR.fields
    for f in fields:
        res_dict[f.name] = getattr(msg, f.name)
    return res_dict


def assign_configs_value(msg, config):
    fields = msg.DESCRIPTOR.fields
    for key in config:
        for f in fields:
            if key == f.name:
                if f.label == 3:
                    getattr(msg, f.name).extend(config[f.name])
                elif f.label == 1 or f.label == 2:
                    setattr(msg, f.name, config[f.name])


def check_configs_key(msg, config, field_name):
    key_list = msg.DESCRIPTOR.fields_by_name.keys()
    for key in config:
        assert key in key_list, "key:{} not in {}".format(key, field_name)


class DistributedJobInfo(object):
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


class DistributedStrategy(object):
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
        self.__lock_attr = True

    def __setattr__(self, key, value):
        if self.__lock_attr and not hasattr(self, key):
            raise TypeError("%s is not a attribute of %s" %
                            (key, self.__class__.__name__))
        object.__setattr__(self, key, value)

    def save_to_prototxt(self, output):
        """
        Serialize current DistributedStrategy to string and save to output file

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.dgc = True
            strategy.recompute = True
            strategy.recompute_configs = {"checkpoint": ["x"]}
            strategy.save_to_prototxt("dist_strategy.prototxt")
        """
        with open(output, "w") as fout:
            fout.write(str(self.strategy))

    def load_from_prototxt(self, pb_file):
        """
        Load from prototxt file for DistributedStrategy initialization

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.load_from_prototxt("dist_strategy.protoxt")
        """
        with open(pb_file, 'r') as f:
            self.strategy = google.protobuf.text_format.Merge(
                str(f.read()), self.strategy)

    @property
    def execution_strategy(self):
        """
        Configure ExecutionStrategy for DistributedStrategy

        Examples:
          .. code-block:: python

            exe_strategy = paddle.fluid.ExecutionStrategy()
            exe_strategy.num_threads = 10
            exe_strategy.num_iteration_per_drop_scope = 10
            exe_strategy.num_iteration_per_run = 10

            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.execution_strategy = exe_strategy
        """
        execution_strategy = paddle.fluid.ExecutionStrategy()
        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(execution_strategy, f.name,
                    getattr(self.strategy.execution_strategy, f.name))
        return execution_strategy

    @execution_strategy.setter
    def execution_strategy(self, strategy):
        fields = self.strategy.execution_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(self.strategy.execution_strategy, f.name,
                    getattr(strategy, f.name))

    @property
    def build_strategy(self):
        """
        Configure BuildStrategy for DistributedStrategy
        Note that the properties of BuildStrategy are valid in DistributedStrategy
        only if the property is non-distributed strategy.

        Examples:
          .. code-block:: python

            build_strategy = paddle.fluid.BuildStrategy()
            build_strategy.enable_sequential_execution = True
            build_strategy.fuse_elewise_add_act_ops = True
            build_strategy.fuse_bn_act_ops = True
            build_strategy.enable_auto_fusion = True
            build_strategy.fuse_relu_depthwise_conv = True
            build_strategy.fuse_broadcast_ops = True
            build_strategy.fuse_all_optimizer_ops = True
            build_strategy.enable_inplace = True
            
            strategy = paddle.distributed.fleet.DistributedStrategy()
            strategy.build_strategy = build_strategy
        """

        build_strategy = paddle.fluid.BuildStrategy()
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            setattr(build_strategy, f.name,
                    getattr(self.strategy.build_strategy, f.name))
        return build_strategy

    @build_strategy.setter
    def build_strategy(self, strategy):
        fields = self.strategy.build_strategy.DESCRIPTOR.fields
        for f in fields:
            if f.label == 1 or f.label == 2:  # optional and required field
                setattr(self.strategy.build_strategy, f.name,
                        getattr(strategy, f.name))
            elif f.label == 3:  # repeated field
                getattr(self.strategy.build_strategy,
                        f.name).extend(getattr(strategy, f.name))

    @property
    def a_sync(self):
        """
        Indicating whether we are using asynchronous stocastic gradient descent updates
        for training. This property is valid when we are using parameter server training, 
        which is implied by setting approperate RoleMaker
        Default value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            role_maker = fleet.PaddleCloudRoleMaker()
            fleet.init(role_maker)

            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True  # by default this is True
            
            # code block for defining loss and local optimizer
            # sgd = fleet.distributed_optimizer(optimizer, strategy)
        """
        return self.strategy.a_sync

    @a_sync.setter
    def a_sync(self, flag):
        if isinstance(flag, bool):
            self.strategy.a_sync = flag
            self.a_sync_configs = {"k_steps": 0}
        else:
            raise ValueError(
                "The type of `flag` is invalid, expected type is bool, but received %s".
                format(type(flag)))

    @property
    def a_sync_configs(self):
        """
        Set a_sync update configurations. In general, asynchronous parameter server
        training has serveral configurable settings that can be configured through
        a dict.

        **Notes**:
            **Detailed arguments for a_sync_configs**
            **k_step**: number of local optimization updates before communication
            **max_merge_var_num**: maximum number of merged gradients before communication
            **send_queue_size**: a buffer size of worker communication
            **independent_recv_thread**: if we are using independent recv thread for communication
            **thread_pool_size**: number of thread pool
            **send_wait_times**: waiting time for sending gradients
            **runtime_split_send_recv**: if we are using Tensor split for send and recv during runtime

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            role_maker = fleet.PaddleCloudRoleMaker()
            fleet.init(role_maker)

            strategy = fleet.DistributedStrategy()
            strategy.a_sync = True  # by default this is True
            configs = {"k_step": 10000, "send_queue_size": 32}
            strategy.a_sync_configs = configs

            # code block for defining loss and local optimizer
            # sgd = fleet.distributed_optimizer(optimizer, strategy)
        """
        return get_msg_dict(self.strategy.a_sync_configs)

    @a_sync_configs.setter
    def a_sync_configs(self, configs):
        check_configs_key(self.strategy.a_sync_configs, configs,
                          "a_sync_configs")
        assign_configs_value(self.strategy.a_sync_configs, configs)

    @property
    def amp(self):
        """
        Indicating whether we are using automatic mixed precision training
        Default Value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.amp = True # by default this is false

        """
        return self.strategy.amp

    @amp.setter
    def amp(self, flag):
        if isinstance(flag, bool):
            self.strategy.amp = flag
        else:
            print("WARNING: amp should have value of bool type")

    @property
    def amp_configs(self):
        return get_msg_dict(self.strategy.amp_configs)

    @amp_configs.setter
    def amp_configs(self, configs):
        check_configs_key(self.strategy.amp_configs, configs, "amp_configs")
        assign_configs_value(self.strategy.amp_configs, configs)

    @property
    def recompute(self):
        """
        Indicating whether we are using forward recomputation for memory optimization
        Default value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.recompute = True
            # suppose x and y are names of checkpoint tensors for recomputation
            strategy.recompute_configs = {"checkpoints": ["x", "y"]}
        """
        return self.strategy.recompute

    @property
    def sync_nccl_allreduce(self):
        """
        Indicating whether we are using synchronized all reduce in each communication thread
        We note that system overhead is usually lower when sync_nccl_allreduce = True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.sync_nccl_allreduce = True
        """
        return self.strategy.sync_nccl_allreduce

    @sync_nccl_allreduce.setter
    def sync_nccl_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_nccl_allreduce = flag
        else:
            print("WARNING: sync_nccl_allreduce should have value of bool type")

    @property
    def use_hierarchical_allreduce(self):
        """
        Indicating whether we are using hierarchical allreduce in collective communication
        Hierarchical allreduce often does allreduce within a certain node group and then do
        allreduce among the leaders of each group

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.use_hierarchical_allreduce = True
        """
        return self.strategy.use_hierarchical_allreduce

    @use_hierarchical_allreduce.setter
    def use_hierarchical_allreduce(self, flag):
        if isinstance(flag, bool):
            self.strategy.use_hierarchical_allreduce = flag
        else:
            print(
                "WARNING: use_hierarchical_allreduce should have value of bool type"
            )

    @property
    def hierarchical_allreduce_inter_nranks(self):
        """
        Number of ranks for low level node groups in hierarchical allreduce
        Default value: number of GPU cards on each single GPU machine

        Example:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.hierarchical_allreduce_inter_nranks = 8
        """
        return self.strategy.hierarchical_allreduce_inter_nranks

    @hierarchical_allreduce_inter_nranks.setter
    def hierarchical_allreduce_inter_nranks(self, value):
        if isinstance(value, int):
            self.strategy.hierarchical_allreduce_inter_nranks = value
        else:
            print(
                "WARNING: hierarchical_allreduce_inter_nranks should have value of int type"
            )

    @property
    def sync_batch_norm(self):
        """
        Indicating whether we are using sync_batch_norm to do synchronous batch normalization among all training nodes.
        
        Default value: False

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.sync_batch_norm = True
        """

        return self.strategy.sync_batch_norm

    @sync_batch_norm.setter
    def sync_batch_norm(self, flag):
        if isinstance(flag, bool):
            self.strategy.sync_batch_norm = flag
        else:
            print("WARNING: sync_batch_norm should have value of bool type")

    @property
    def fuse_all_reduce_ops(self):
        """
        Indicating whether we are using fuse_all_reduce_ops for gradient fusion during backward phase of training
        Default value: True

        Examples:
          .. code-block:: python

            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fuse_all_reduce_ops = False
        """
        return self.strategy.fuse_all_reduce_ops

    @fuse_all_reduce_ops.setter
    def fuse_all_reduce_ops(self, flag):
        if isinstance(flag, bool):
            self.strategy.fuse_all_reduce_ops = flag
        else:
            print("WARNING: fuse_all_reduce_ops should have value of bool type")

    @property
    def fuse_grad_size_in_MB(self):
        """
        Specifying the size of gradient to fuse in Mega-Bytes

        Default value: 32

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.fuse_grad_size_in_MB = 50
        """
        return self.strategy.fuse_grad_size_in_MB

    @fuse_grad_size_in_MB.setter
    def fuse_grad_size_in_MB(self, value):
        if isinstance(value, int):
            self.strategy.fuse_grad_size_in_MB = value
        else:
            print("WARNING: fuse_grad_size_in_MB should have value of int type")

    @property
    def _fuse_grad_size_in_TFLOPS(self):
        return self.strategy.fuse_grad_size_in_TFLOPS

    @_fuse_grad_size_in_TFLOPS.setter
    def _fuse_grad_size_in_TFLOPS(self, value):
        if isinstance(value, float):
            self.strategy.fuse_grad_size_in_TFLOPS = value
        else:
            print(
                "WARNING: fuse_grad_size_in_TFLOPS should have value of float type"
            )

    @property
    def nccl_comm_num(self):
        """
        Specifying the number of NCCL communicator

        Default value: 1

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.nccl_comm_num = 2
        """

        return self.strategy.nccl_comm_num

    @nccl_comm_num.setter
    def nccl_comm_num(self, value):
        if isinstance(value, int):
            self.strategy.nccl_comm_num = value
        else:
            print("WARNING: nccl_comm_num should have value of int type")

    @recompute.setter
    def recompute(self, flag):
        if isinstance(flag, bool):
            self.strategy.recompute = flag
        else:
            print("WARNING: recompute should have value of bool type")

    @property
    def recompute_configs(self):
        """
        Set recompute configurations. In general, the recompute strategy of current
        implementation should have some manually assign checkpoints

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.recompute = True
            strategy.recompute_configs = {"checkpionts": ["x", "y"]}

        """
        return get_msg_dict(self.strategy.recompute_configs)

    @recompute_configs.setter
    def recompute_configs(self, configs):
        check_configs_key(self.strategy.recompute_configs, configs,
                          "checkpoint_configs")
        assign_configs_value(self.strategy.recompute_configs, configs)

    @property
    def pipeline(self):
        """
        Indicating whether we are using pipeline parallelism for distributed training.
        Current implementation mainly focus on single GPU machine pipeline parallelism and
        data parallelism across GPU machine. The pipeline information is indicated through
        device_guard information in user-defined program.

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True

        """
        return self.strategy.pipeline

    @pipeline.setter
    def pipeline(self, flag):
        if isinstance(flag, bool):
            self.strategy.pipeline = flag
        else:
            print("WARNING: pipeline should have value of bool type")

    @property
    def pipeline_configs(self):
        """
        Set pipeline parallelism configurations. In pipeline parallelism,
        different parts of neural networks are running on different GPUS.
        There are Tensor queue buffer between each pair of neighborhood GPUS 
        that are responsible for synchronizing hidden Tensor results between
        GPUs. Pipeline parallelism consists of serveral producer-consumer style
        hardware pairs, such as GPU-GPU, CPU-GPU, GPU-XPU. The best way to speedup
        pipeline parallelism is to make the size of Tensor in Tensor queue smaller, 
        so that we will have a faster producer for downstream consumers.

        **Notes**:
            **Detailed arguments for pipeline_configs**
            **micro_batch**: the number of small batches in each user defined batch

        Examples:
          .. code-block:: python
        
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.pipeline = True
            strategy.pipeline_configs = {"micro_batch": 12}

        """

        return get_msg_dict(self.strategy.pipeline_configs)

    @pipeline_configs.setter
    def pipeline_configs(self, configs):
        check_configs_key(self.strategy.pipeline_configs, configs,
                          "pipeline_configs")
        assign_configs_value(self.strategy.pipeline_configs, configs)

    @property
    def localsgd(self):
        return self.strategy.localsgd

    @localsgd.setter
    def localsgd(self, flag):
        if isinstance(flag, bool):
            self.strategy.localsgd = flag
        else:
            print("WARNING: localsgd should have value of bool type")

    @property
    def localsgd_configs(self):
        return get_msg_dict(self.strategy.localsgd_configs)

    @localsgd_configs.setter
    def localsgd_configs(self, configs):
        check_configs_key(self.strategy.localsgd_configs, configs,
                          "localsgd_configs")
        assign_configs_value(self.strategy.localsgd_configs, configs)

    @property
    def dgc(self):
        return self.strategy.dgc

    @dgc.setter
    def dgc(self, flag):
        if isinstance(flag, bool):
            self.strategy.dgc = flag
        else:
            print("WARNING: dgc should have value of bool type")

    @property
    def dgc_configs(self):
        return get_msg_dict(self.strategy.dgc_configs)

    @dgc_configs.setter
    def dgc_configs(self, configs):
        check_configs_key(self.strategy.dgc_configs, configs, "dgc_configs")
        assign_configs_value(self.strategy.dgc_configs, configs)

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
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
        """
        return self.strategy.gradient_merge

    @gradient_merge.setter
    def gradient_merge(self, flag):
        if isinstance(flag, bool):
            self.strategy.gradient_merge = flag
        else:
            print("WARNING: gradient_merge should have value of bool type")

    @property
    def gradient_merge_configs(self):
        """
        the key-value configs of distribute_strategy
        Keys: 
            k_steps (int): the update period of the parameters
            avg (bool): whether to average the gradients of each mini-batch,
                the default value is `True`
        Example:
            import paddle.distributed.fleet as fleet
            strategy = fleet.DistributedStrategy()
            strategy.gradient_merge = True
            strategy.gradient_merge_configs = {"k_steps": 4, "avg": True}
        """
        return get_msg_dict(self.strategy.gradient_merge_configs)

    @gradient_merge_configs.setter
    def gradient_merge_configs(self, configs):
        check_configs_key(self.strategy.gradient_merge_configs, configs,
                          "gradient_configs")
        assign_configs_value(self.strategy.gradient_merge_configs, configs)

    @property
    def lars(self):
        return self.strategy.lars

    @lars.setter
    def lars(self, flag):
        if isinstance(flag, bool):
            self.strategy.lars = flag
        else:
            print("WARNING: lars should have value of bool type")

    @property
    def lars_configs(self):
        return get_msg_dict(self.strategy.lars_configs)

    @lars_configs.setter
    def lars_configs(self, configs):
        check_configs_key(self.strategy.lars_configs, configs, "lars_configs")
        assign_configs_value(self.strategy.lars_configs, configs)

    @property
    def lamb(self):
        return self.strategy.lamb

    @lamb.setter
    def lamb(self, flag):
        if isinstance(flag, bool):
            self.strategy.lamb = flag
        else:
            print("WARNING: lamb should have value of bool type")

    @property
    def lamb_configs(self):
        return get_msg_dict(self.strategy.lamb_configs)

    @lamb_configs.setter
    def lamb_configs(self, configs):
        check_configs_key(self.strategy.lamb_configs, configs, "lamb_configs")
        assign_configs_value(self.strategy.lamb_configs, configs)

    @property
    def elastic(self):
        return self.strategy.elastic

    @elastic.setter
    def elastic(self, flag):
        if isinstance(flag, bool):
            self.strategy.elastic = flag
        else:
            print("WARNING: elastic should have value of bool type")

    @property
    def auto(self):
        return self.strategy.auto

    @auto.setter
    def auto(self, flag):
        if isinstance(flag, bool):
            self.strategy.auto = flag
        else:
            print("WARNING: auto should have value of bool type")

    def __repr__(self):
        fields = self.strategy.DESCRIPTOR.fields
        for f in fields:
            print("{}: {}".format(f.name, f.default_value))
        return str(self.strategy)

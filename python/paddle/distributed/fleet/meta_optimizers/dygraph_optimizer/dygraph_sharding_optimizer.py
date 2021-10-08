# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

######
from functools import reduce

import paddle
from paddle import framework
from ...utils.log_util import logger


def _is_trainable(param: paddle.Tensor) -> bool:
    return not param.stop_gradient


class DygraphShardingOptimizer(object):
    """
    A wrapper for Sharding Optimizer in Dygraph. 

    .. warning: DygraphShardingOptimizer is experimental and subject to change.

    .. ZeRO: https://arxiv.org/abs/1910.02054

    """

    # TODO (JZ-LIANG) 
    # TO support following featrues in future:
    # 1. fused update parameter sync
    # 2. parameters_groups
    # 3. dynamic trainable params, which is the case bewteen pretraining and finetuning
    # 4. option to choose fuse comm (more GPU MEM need) or un-fuse comm

    def __init__(
            self,
            hcg,
            user_defined_strategy,
            params,
            inner_optimizer_class,
            reduce_bucket_size=500000000,
            overlap_comm=False,
            **inner_optimizer_kargs, ):

        if not isinstance(params, list):
            raise TypeError(
                "`parameters` argument given to the DygraphShardingOptimizer should be "
                "an iterable of paddle Tensors, but got argument type is `{}`.".
                format(type(params)))
        self._parameter_list = params
        self._reference_is_trainable_params = list(
            map(_is_trainable, self._parameter_list))

        self._inner_optimizer_class = inner_optimizer_class
        self._inner_optimizer_kargs = inner_optimizer_kargs

        # sharding parallel information
        # TODO better way to get the hcg & user_defined_strategy
        self._hcg = hcg
        self._user_defined_strategy = user_defined_strategy
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        # logic partitioning
        self._build_sharding_mapping()

        # actually create opt ops
        self._buid_inner_optimizer()

        #################################################################
        ################# Zero Stage 2: Gradient Sharding ###############
        #################################################################

        # init reduction stream
        self._reduction_stream = paddle.device.cuda.Stream()

        # reduce bucket grads and gc
        self._elements_in_grad_bucket = 0
        self._grad_bucket_size = reduce_bucket_size
        self._extra_large_param_to_reduce = None
        self._grads_in_grad_bucket = []
        self._params_in_grad_bucket = []

        # store whether param is in this rank
        self._is_param_in_current_rank = {}
        self._params_in_current_rank = []
        self._params_not_in_current_rank = []
        for param in _parameter_list:
            if self._param2rank[param.name] == self._sharding_rank:
                self._is_param_in_current_rank[param.name] = True
                self._params_in_current_rank.append(param)
            else:
                self._is_param_in_current_rank[param.name] = False
                self._params_not_in_current_rank.append(param)

        # recording reduction of params
        self._params_already_reduced = {}
        for param in self._parameter_list:
            self._params_already_reduced[param.name] = False

        # use 2 overlapping cuda streams(default stream and reduction stream)
        self._overlap_comm = overlap_comm

        # register grad reduction and gc hooks
        self._hook_removers = []
        self._register_hooks_for_grad_reduction_and_gc()

        #################################################################
        ################# Zero Stage 2 finishes #########################
        #################################################################

    def clear_grad(self):
        """
        should clear grad for all parameters in model
        """
        for p in self._parameter_list:
            if not p.stop_gradient:
                p.clear_gradient()

    def _build_sharding_mapping(self):

        self._rank2params = self._partition_parameters()
        self._param2rank = self._map_param_to_rank()

    def _partition_parameters(self):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List] 
        """
        # TODO(JZ-LIANG) support multiple partition methods
        # method1: greedy even but unorder
        # method2: roughly even with oreder

        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size
        for param in self._parameter_list:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape)
            assert numel > 0, "param [{}] should larger than 0, but it is [{}]".format(
                param.name, numel)
            sizes[rank] += numel

        return mapping

    def _map_param_to_rank(self):
        """
        mapping parameters to the shard which holds it.

        Return:
        Dict[str, int] 
        """
        mapping = {}
        for rank, params in self._rank2params.items():
            for param in params:
                mapping[param.name] = rank
        return mapping

    def _buid_inner_optimizer(self):
        # we rely on the inner opt to determine whether a parameter is stop_gradient or not:
        # create moment
        # update related ops: clip, regular, opt  
        self._inner_optimizer = self._inner_optimizer_class(
            parameters=self._rank2params[self._sharding_rank],
            **self._inner_optimizer_kargs)

    def _sharding_sync_parameters(self):
        """
        sync parameter across sharding group
        """
        # TODO speed up this functional

        logger.debug("sharding start sync parameters")
        with framework.no_grad():
            # TODO detach not need (?)
            for rank, params in self._rank2params.items():
                for param in params:
                    paddle.distributed.broadcast(
                        param,
                        # the collective API need src rank to be the global rank id 
                        # instead of the relative logic rank id within group 
                        src=self._hcg.get_sharding_parallel_group().ranks[rank],
                        group=self._hcg.get_sharding_parallel_group(),
                        use_calc_stream=True)

    def _update_trainable(self):
        """
        allow user to update trainable parameters list during training
        """
        raise NotImplementedError

    def _pre_update(self):
        self._backward_pass_grad_reduction_epilogue()

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameters=None,
                 no_grad_set=None):

        # NOTE in dygraph mode, the only different between step and minimize is that minimize 
        # allow user to customize the parameters for updating on each step

        # do something before updating params
        self._pre_update()

        input_param_names = set([param.name for param in parameters])
        parameters = list(
            filter(lambda x: x.name in input_param_names, self._rank2params[
                self._sharding_rank]))
        result = self._inner_optimizer.minimize(loss, startup_program,
                                                parameters, no_grad_set)

        # sync parameters accross sharding ranks
        self._sharding_sync_parameters()

        return result

    def step(self):
        # TODO Check whether the model trainable param changed and update state accordingly

        # do something before updating params
        self._pre_update()

        # actually updating
        self._inner_optimizer.step()

        # sync parameters accross sharding ranks
        self._sharding_sync_parameters()

    # TODO is it a good way to make _grad_clip a property
    @property
    def _grad_clip(self):
        assert self._inner_optimizer is not None, "inner opt of sharding is not initiliazed."
        return self._inner_optimizer._grad_clip

    def __getattr__(self, item):
        return getattr(self._inner_optimizer, item)

    #################################################################
    ################# Zero Stage 2: Gradient Sharding ###############
    #################################################################

    # when backward pass is finished,
    # use this epilogue to reduce and GC any remaining grads
    def _backward_pass_grad_reduction_epilogue(self):
        # reduce remaining grads in bucket
        self._reduce_grads_immediately_and_gc()

        if self._overlap_comm:
            # sync remaining reduce kernels in stream
            paddle.device.cuda.synchronize()
            # GC grads reduced by these kernels
            self._clear_previous_reduced_grads()

        # reset reduction state of each param
        for param in self._parameter_list:
            self._params_already_reduced[param.name] = False

    # create hook for each param when its grad is computed
    def _register_hooks_for_grad_reduction_and_gc(self):
        for param in self._parameter_list:
            if param.requires_grad and not param.stop_gradient:
                # definition of grad reduction closure.
                # the input of this closure (current gradient) is needed
                # because when hook is being executed, param.grad is None
                # only after hook finished its execution will param.grad be set
                def reduce_grad_and_gc_closure(grad):
                    self._reduce_grad_buckets_and_gc(param, grad)
                hook_remover = param.register_hook(reduce_grad_and_gc_closure)
                self._hook_removers.append(hook_remover)

    # currently useless
    def _remove_hooks_for_grad_reduction_and_gc(self):
        for hook_remover in self._hook_removers:
            hook_remover.remove()

    # use buckets for gradient reduction
    def _reduce_grad_buckets_and_gc(self, param, grad=None):
        # if grad bucket is full
        if self._elements_in_grad_bucket + param.numel() > self._grad_bucket_size:
            self._reduce_grads_immediately_and_gc()

        # if a single param is larger than bucket size
        if param.numel() > self._grad_bucket_size:
            self._extra_large_param_to_reduce = param

        self._elements_in_grad_bucket += param.numel()

        if grad is None:    # if entered by epilogue
            self._grads_in_grad_bucket.append(param.grad)
        else:               # if entered by hooks
            self._grads_in_grad_bucket.append(grad)

        self._params_in_grad_bucket.append(param)

    # reduce gradients immediately when grad bucket is full or backward pass is finished.
    def _reduce_grads_immediately_and_gc(self):
        if self._overlap_comm:
            # wait for previous reduction kernels to complete
            paddle.device.cuda.synchronize()

            # clear grads in last reduction
            self._clear_previous_reduced_grads()

            # use reduction stream for reduction, default stream for calculation
            stream = self._reduction_stream
        else:
            stream = paddle.device.cuda.current_stream()

        with paddle.device.cuda.stream_guard(stream):
            # step 1: reduction
            self._buffered_reduce(self._grads_in_grad_bucket, self._elements_in_grad_bucket)

            # step 2: gradient Garbage Collection
            for param in self._params_in_grad_bucket:
                # one param cannot be reduced twice
                assert self._params_already_reduced[param.name] is False
                self._params_already_reduced[param.name] = True

                if not self._is_param_in_current_rank[param.name]:
                    if self._overlap_comm:
                        # GC grads of other sharding ranks during the next reduction
                        # to avoid clearing them before the reduction is complete.
                        if self._previous_reduced_grads is None:
                            self._previous_reduced_grads = []
                        self._previous_reduced_grads.append(param)
                    else:
                        param.grad = None

        self._grads_in_grad_bucket = []
        self._params_in_grad_bucket = []
        self._elements_in_grad_bucket = 0

    # reduce communicator for averaged gradients
    def _buffered_reduce(self, grads, params):
        #TODO: fused param reduce
        for grad, param, _ in grads, zip(params):
            dst_rank = self._param2rank[param.name]
            tensor_to_reduce = grad.div_(1. / self._sharding_world_size)
            paddle.distributed.reduce(tensor_to_reduce,
                        self._hcg.get_sharding_parallel_group().ranks[dst_rank],
                        group=self._hcg.get_sharding_parallel_group())
            grad.copy_(tensor_to_reduce)

    # TODO: fuse multiple tensors for the same rank
    def _buffered_reduce_rank_aggregation(self, grads, params):
        # init data structures
        params_group_by_rank = {}
        for rank in range(self._sharding_world_size):
            params_group_by_rank[rank] = []

        # group params by rank
        for param in params:
            rank = self._param2rank[param.name]
            params_group_by_rank[rank].append(param)

        # TODO: Reduce Grouped Params
        raise NotImplementedError("rank fuse not implemented!")

    # gc gradients
    def _clear_previous_reduced_grads(self):
        if self._previous_reduced_grads is not None:
            for param in self._previous_reduced_grads:
                param.grad = None
            self._previous_reduced_grads = None

    #################################################################
    ################# Zero Stage 2 finishes #########################
    #################################################################
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""
A nn.Module wrapper to go with a Sharded Optimizer in order to handle targeted gradient
reduction automatically.
"""
import os
from collections import deque
import contextlib
import functools
from itertools import chain
import logging
import time

import paddle
from paddle import nn
# from torch.autograd import Variable
# import torch.autograd.profiler as profiler
import paddle.distributed as dist

from .bucket import GradBucket
from .oss import OSS
from .sharding_utils import Workhandle, GpuInfo


def _trainable(param):
    return param.trainable


class ShardingStage2(nn.Layer):
    """ 
    A wrapper for Sharding Stage2 Layer in Dygraph. 

    .. warning: ShardingStage2 encapsulates the layer strategy and integrates it into the nn.Layer.

    .. ZeRO: https://arxiv.org/pdf/1910.02054.pdf
    """

    # TODO (Baibaifan) 
    # TO support following featrues in future:
    # 1. Unified memory for param and param.grad to InternalStorage.
    # 2. Divide param.grad according to rank to centrally apply for and release GPU memory.
    # 3. Dynamically adjust training parameters and models。
    # 4. Support offload function.
    # 5. Support the establishment of independent communication groups.

    def __init__(
            self,
            layer,
            sharding_optimizer,
            group,
            broadcast_buffers=False,
            pertrain_sync_models=True,
            grad_buffer_size=2**24,  #8MB 16MB
            auto_refresh_trainable=True,
            device="gpu",
            reduce_fp16=False):
        super().__init__()

        # This field needs to be exposed to insure interface parity with DDP
        self.layer = layer

        self._sharded_optimizers = [sharded_optimizer] if not isinstance(
            sharded_optimizer, list) else sharded_optimizer
        self._enable_broadcast_buffers = broadcast_buffers
        self._auto_refresh_trainable = auto_refresh_trainable
        # maybe have errors in paddle
        self._reduce_fp16 = reduce_fp16
        if reduce_buffer_size > 0 and reduce_fp16:
            self._reduce_fp16 = False
            logging.warning(
                "fp16 gradient reduction is not compatible with reduction buffers, which are requested. fp16 grad reduction is deactivated."
            )

        # Handle a no_sync() context which prevents the gradient synchronization,
        # accumulate in place
        self._should_accumulate_grads = False
        self._accumulate_grads_flipped = False

        # Communication related attributes
        assert process_group is not None, "Distributed communication group is must be gived"
        self._process_group = process_group
        # self._backend = process_group.backend
        self._world_size_scaling = 1.0 / self._process_group.nranks  # > 0
        self._rank = process_group.rank
        dev_id = self._rank
        self._global_root_rank = 0  # picking rank 0 as the reference
        self._local_to_global_rank = self._process_group.ranks
        self._default_device = use_device

        # Scafolding to be able to reduce the grads during the BW pass
        # several optimizers can be present each working on seperate parameter set which is spread across multiple ranks

        # - we build an iterator which goes through all the parameters involved globally
        self._all_params = list(
            chain(* [
                sum([sum(p, []) for p in optim.per_device_params.values()], [])
                for optim in self._sharded_optimizers
            ]))
        self._trainable_params = []
        self._grad_to_be_reduced = []
        self._trainable_param_to_rank = {}
        self._reference_trainable_mask = list(map(_trainable, self._all_params))

        # - setup buckets and tensor views
        model_size = sum([p.numel() for p in self.layer.parameters()]).item()
        self._buffer_max_size = min(reduce_buffer_size, model_size)

        assert self._process_group.nranks > 1, "Training is not really distributed, single rank. Deactivating buckets"

        logging.info(
            "ShardedDDP bucket size: {:.2f}M parameters, model size {:.2f}M parameters".
            format(self._buffer_max_size / 2**20, model_size / 2**20))
        self._use_buckets = self._buffer_max_size > 0

        self._buckets = {}  # {device: {rank: GradBucket}}
        self._should_bucket_grad = []
        self._bucket_list = []

        # - setup backward hooks which will be called by Paddle's autograd in due time
        self._grad_accs = []
        self._grad_hooks = []
        self._manual_reduce = []

        # passing a handle to torch.nn.SyncBatchNorm layer(paddle undetermined)
        # self._passing_sync_batchnorm_handle(self.layer)

        # Make sure that all ranks start with the same model
        if sync_models_at_startup:
            self._sync_params_and_buffers()

        self._work_handles = deque()
        self._bucket_flush_callback_set = False

    def forward(self, *inputs, **kwargs):
        """
        Module forward pass, handles any DDP-specific work in the background. Primes the
        backward pass for gradient reduction to the proper ranks.
        """

        # Deferred initialization, or change detection
        needs_setup = len(self._grad_hooks) == 0 and self.training

        if self._auto_refresh_trainable:
            needs_setup |= self._detect_train_change()

        if needs_setup:
            self.refresh_trainable()
        else:
            self.build_bucket_buffer()

        if self._enable_broadcast_buffers:
            # NCCL communications are on a different stream, needs to be blocking
            # for the subsequent FW to be correct
            self.sync_buffers(blocking=True)

        # Reset all the grad reduce and bucket state flags
        self._clear_counters()

        # Normal FW on the base model
        return self.layer(*inputs, **kwargs)

    def to(self, device, dtype=None, blocking=True):
        """
        Moves and/or casts the parameters and buffers.

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point desired :attr:`dtype` s. In addition, this method will
        only cast the floating point parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        .. note::
            This method modifies the module in-place.

        .. warning:
            Device changes are not supported, and this will raise an exception. The issue in that case is not
            really ShardedDDP, but OSS which will not be aware of the device change, and whose buffers will be
            in a broken state.

        Arguments:
            device (:class:`paddle.device`): the desired device of the parameters and buffers in this module.
            dtype (:class:`paddle.dtype`): the desired floating point type of the floating point parameters and buffers.
            blocking (bool): make it a synchronous call.

        Returns:
            Module: self.
        """

        assert isinstance(device, str), "Device must be type str"
        assert (
            device is None or len(self._buckets.keys()) == 0 or
            device in self._buckets.keys()
        ), "Changing devices is not supported, because this would break OSSs state"

        assert (
            len(self._buckets.keys()) < 2
        ), "Several devices specified to begin with, incompatible with setting a single device here"

        self.layer.to(device=device, dtype=dtype, blocking=blocking)

        # Re-build the buckets, hooks, etc..
        self.refresh_trainable()

    def refresh_trainable(self):
        """ If the module trainability has changed, update all the assumptions """

        # Make sure that this is not done while gradients are waiting to be reduced (if no_sync context for instance)
        if functools.reduce(lambda x, y: x or y, self._grad_to_be_reduced,
                            False):
            logging.warning(
                "Grads waiting to be reduced. If this is on purpose (grad accumulation), please use a no_sync() context"
            )

        self._trainable_params = list(
            filter(lambda x: x.trainable, self._all_params))
        self._trainable_params.sort(key=lambda x: x.numel())

        self._trainable_param_to_rank = {}
        for optim in self._sharded_optimizers:
            # OSS may need to change the communication pattern
            if len(optim.buckets.keys()) == 0:
                optim.refresh_trainable()

            # Update ShardedDDP given the new partitions
            for (device_per_rank_params) in optim._per_device_params.values(
            ):  # all the params on this device (inc all ranks)
                for device_params in device_per_rank_params:
                    for param in filter(lambda x: x.trainable, device_params):
                        self._trainable_param_to_rank[
                            param] = optim.param_to_rank[param]

            self._setup_bucket_strategy()
            self._setup_backward_hooks()

    @paddle.no_grad()
    def sync_buffers(self, blocking=False):
        """
        Sync all the param buffers in between ranks (including for instance batch norm statistics).

        Arguments:
            blocking (bool): wait for the operation to conclude.
        """

        work_handles = []

        for buffer in self.layer.buffers(include_sublayers=True):
            work_handles.append(
                dist.broadcast(
                    buffer,
                    self._global_root_rank,
                    self._process_group,
                    use_calc_stream=True))

        if blocking:
            dist.wait(
                tensor=buffer, group=self._process_group, use_calc_stream=True)

    def zero_grad(self, set_to_none=False):
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """

        for index, trainable_param in enumerate(self._trainable_params):
            if set_to_none and (len(self._should_bucket_grad) == 0 or
                                not self._should_bucket_grad[index]):
                trainable_param._clear_gradient()
            elif trainable_param.grad is not None:
                trainable_param.grad.zero_()

        for bucket in self._bucket_list:
            bucket.zero()

    def __getattr__(self, name):
        """Forward missing attributes to wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.layer, name)

    @contextlib.contextmanager
    def no_sync(self):
        """A context manager to disable gradient synchronization."""
        old_should_accumulate_grads = self._should_accumulate_grads
        self._should_accumulate_grads = True
        yield
        self._accumulate_grads_flipped = self._should_accumulate_grads != old_should_accumulate_grads
        self._should_accumulate_grads = old_should_accumulate_grads

    @paddle.no_grad()
    def _clear_counters(self):
        """Reset all the grad reduce and call counters"""
        if self.training:
            self._grad_to_be_reduced = [True for _ in self._trainable_params]
        self._bucket_flush_callback_set = False

        if self._use_buckets:
            for bucket in self._bucket_list:
                bucket.reset_checked_in()

        if not self._should_accumulate_grads:
            self._accumulate_grads_flipped = False

    def _get_reduce_fn(self, index, param, dst_rank):
        """
        Two possible backward hooks for a given parameter: either directly reduce to the appropriate rank,
        or contribute to a bucket and reduce when the bucket is full.

        Either way a delayed action is necessary and is passed as a callback.
        """

        if not self._use_buckets or not self._should_bucket_grad[index]:
            # Direct reduction
            @paddle.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not alter status flags
                if not self._should_accumulate_grads and self._grad_to_be_reduced[
                        index]:
                    assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                    # maybe have errors in paddle
                    # if not self._bucket_flush_callback_set:
                    #     Variable._execution_engine.queue_callback(self._flush_reduce_calls)
                    #     self._bucket_flush_callback_set = True

                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False
                    paddle.multiply(param.grad,
                                    paddle.to_tensor(
                                        [self._world_size_scaling]))

                    # maybe have errors in paddle
                    if self._reduce_fp16:
                        paddle.cast(param.grad, paddle.float16)

                    # Future work includes clearing up the buffer if possible
                    def cleanup():
                        if dst_rank != self._rank:
                            param._clear_gradient()
                        else:
                            assert param.grad is not None
                            paddle.cast(param.grad, param.dtype)

                    # Async reduce for this buffer, log the future
                    self._work_handles.append(
                        Workhandle(
                            handle=dist.reduce(
                                tensor=param.grad,
                                dst=dst_rank,
                                group=self._process_group,
                                use_calc_stream=True),
                            callback=cleanup))
                    dist.wait(
                        tensor=param.grad,
                        group=self._process_group,
                        use_calc_stream=True)

                    # Opportunistically try to empty the queue, free memory
                    # maybe have errors async in paddle
                    self._try_consume_work_handle()

        else:

            @paddle.no_grad()
            def reduce(*_):
                # Skip gradient reduction, do not alter status flags

                if not self._should_accumulate_grads and self._grad_to_be_reduced[
                        index]:
                    assert param.grad is not None, "Reducing gradients during backward pass, cannot be None"

                    # Make sure that this is not fired twice
                    self._grad_to_be_reduced[index] = False
                    bucket = self._buckets[self._default_device][dst_rank]
                    bucket.params_checked_in += 1

                    if bucket.all_checked_in:
                        assert bucket.buffer is not None

                        # Normalize the bucket in one go
                        paddle.multiply(bucket.buffer,
                                        paddle.to_tensor(
                                            [self._world_size_scaling]))

                        # Future work includes clearing up the buffer if possible
                        def cleanup():
                            if dst_rank != self._rank:
                                for p in bucket._params:
                                    p._clear_gradient()
                                    print("bbbbbbbbbb", p.grad)
                                print("ddddddddddddd")
                                time.sleep(10)
                                print(bucket.buffer.use_count())
                                bucket.buffer.value().get_tensor()._clear()
                                print(bucket.buffer.use_count())
                                # bucket.buffer = None
                                print("ssssssssssssss")
                                time.sleep(10)

                        # Reduce the bucket
                        bucket.sent = True
                        self._work_handles.append(
                            Workhandle(
                                handle=dist.reduce(
                                    tensor=bucket.buffer,
                                    dst=bucket.destination,
                                    group=self._process_group,
                                    use_calc_stream=True),
                                callback=cleanup))
                        dist.wait(
                            tensor=bucket.buffer,
                            group=self._process_group,
                            use_calc_stream=True)

                    # Opportunistically try to empty the queue
                    self._try_consume_work_handle()

        return reduce

    def _setup_backward_hooks(self):
        """
        Attach a reduce function to each grad-requiring parameter.
        This makes the gradient reduction automatic whenever there's a backward pass
        """

        # Detach possible pre-existing hooks
        while len(self._grad_hooks) > 0:
            self._grad_hooks.pop().remove()

        # Go through the parameters, attach the hook
        self._grad_accs = []
        self._manual_reduce = []
        if not self.training:
            return

        for index, param in enumerate(self._trainable_params):
            dst_rank = self._trainable_param_to_rank[param]

            reduce_function = self._get_reduce_fn(index, param, dst_rank)

            self._grad_hooks.append(
                param._register_backward_hook(reduce_function))
            self._manual_reduce.append(reduce_function)

    @paddle.no_grad()
    def _sync_params_and_buffers(self):
        """
        Sync the complete model states in between the ranks
        """

        work_handles = []

        for t in self.layer.state_dict().values():
            work_handles.append(
                dist.broadcast(
                    t,
                    src=self._global_root_rank,
                    group=self._process_group,
                    use_calc_stream=True))
            dist.wait(tensor=t, group=self._process_group, use_calc_stream=True)

    def _setup_bucket_strategy(self):
        """Devise a bucketing strategy on a per-rank ownership level.
        These buckets will not be sharded, since the gradients would be re-allocated during the backward in that case.
        This method can be a slow for big models, but it it not typically called often (not for every forward for instance)
        """

        if not self._use_buckets:
            return

        # Devise the bucketing strategy. Parameters are already sorted, in that:
        # - these are only the trainable parameters, so they should produce grads
        # - they are sorted by increasing size
        self._buckets = {}
        self._should_bucket_grad = [False for _ in self._trainable_params]

        for i, param in enumerate(self._trainable_params):
            dst_rank = self._trainable_param_to_rank[param]

            if self._default_device not in self._buckets.keys():
                self._buckets[self._default_device] = {}

            if dst_rank not in self._buckets[self._default_device].keys():
                self._buckets[self._default_device][dst_rank] = GradBucket(
                    self._buffer_max_size,
                    dtype=param.dtype,
                    device=self._default_device,
                    destination=dst_rank)

            # Criteria to decide whether this parameter is to be bucketed or not:
            # - enough room in the bucket
            if self._buckets[self._default_device][dst_rank].can_add_grad_view(
                    param):
                self._buckets[self._default_device][dst_rank].add_grad(param)
                self._should_bucket_grad[i] = True

        self._bucket_list = list(
            chain(* [
                self._buckets[device].values()
                for device in self._buckets.keys()
            ]))

        # Resize the buckets to remove lost space in the end
        for bucket in self._bucket_list:
            bucket.shrink()

    # maybe have errors in paddle
    def _try_consume_work_handle(self):
        """Try to consume the oldest future. This is non blocking, if not ready we'll pass"""
        # while len(self._work_handles) > 0 and self._work_handles[0].handle.is_completed():
        while len(self._work_handles) > 0:
            work_handle = self._work_handles.popleft()
            if work_handle.callback is not None:
                print("1111111111111111111")
                work_handle.callback()

    def _detect_train_change(self):
        # Optionally check whether the trainable parameters have changed
        trainable_mask = list(map(_trainable, self._all_params))

        # - one or more parameters trainability changed
        trainability_changed = trainable_mask != self._reference_trainable_mask

        # - the whole model is not trainable but we still have grad hooks
        trainability_changed |= not self.training and len(self._grad_hooks) > 0

        if trainability_changed:
            logging.warning(
                "ShardedDDP detected that the trainable params changed, either because of eval/train mode or parameter freezing/unfreeze."
            )
            self._reference_trainable_mask = trainable_mask

        return trainability_changed

    def build_bucket_buffer(self):
        for dst_rank, bucket in self._buckets[self._default_device].items():
            bucket._fill = 0
            bucket._is_collapsed = True
            bucket.rebuild()

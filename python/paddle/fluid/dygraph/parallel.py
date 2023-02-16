# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except jin compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import warnings
from collections import OrderedDict
import itertools
import warnings
from contextlib import contextmanager

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.dygraph import layers
from paddle.fluid.dygraph import parallel_helper
from paddle.fluid.dygraph import to_variable, no_grad
from paddle.utils import deprecated
from ..layers import collective
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid.framework import (
    ParamBase,
    in_dygraph_mode,
)

__all__ = ["DataParallel"]

ParallelStrategy = core.ParallelStrategy


def _build_default_parallel_strategy():
    strategy = ParallelStrategy()
    strategy.nranks = paddle.distributed.ParallelEnv().nranks
    strategy.local_rank = paddle.distributed.ParallelEnv().local_rank
    strategy.trainer_endpoints = (
        paddle.distributed.ParallelEnv().trainer_endpoints
    )
    strategy.current_endpoint = (
        paddle.distributed.ParallelEnv().current_endpoint
    )
    return strategy


def _coalesce_tensors(var_groups):
    from ..layers import nn

    coalesced_grads_and_grad_vars = []
    for group_id, grad_vars in var_groups.items():
        flattened_vars = []
        g_var_shapes = []
        for g_var in grad_vars:
            g_var_shapes.append(g_var.shape)
            flattened_vars.append(
                paddle.reshape(x=g_var, shape=[np.prod(g_var.shape)])
            )
        coalesced_grad = paddle.concat(flattened_vars)
        coalesced_grads_and_grad_vars.append(
            [coalesced_grad, grad_vars, g_var_shapes]
        )
    return coalesced_grads_and_grad_vars


@framework.dygraph_only
def _reshape_inplace(x, shape):
    x_shape = framework._varbase_creator(dtype=x.dtype)
    framework._dygraph_tracer().trace_op(
        type="reshape2",
        inputs={'X': x},
        outputs={'Out': x, 'XShape': x_shape},
        attrs={'shape': shape},
    )


@framework.dygraph_only
def _split_tensors(coalesced_grads_and_grad_vars):
    if in_dygraph_mode():
        for (
            coalesced_grad,
            origin_grad_vars,
            grad_shapes,
        ) in coalesced_grads_and_grad_vars:
            grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
            attrs = ()
            attrs += ('sections', grad_var_len)
            attrs += ('axis', 0)
            _legacy_C_ops.split(coalesced_grad, origin_grad_vars, *attrs)
            for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
                g_var.reshape_(shape=g_shape)
                assert g_var.shape == g_shape


def scale_loss(loss):
    # TODO(liuyuhui) Currently only for xpu. Will be removed in the future.
    if not paddle.distributed.ParallelEnv().world_size > 1:
        return loss

    loss_scale = to_variable(
        np.array([paddle.distributed.ParallelEnv().world_size]).astype(
            "float32"
        )
    )
    loss_scale.stop_gradient = True
    scaled_loss = loss / loss_scale
    return scaled_loss


@imperative_base.no_grad
@framework.dygraph_only
def build_groups(vars, group_size):
    group_idx = 0
    memory_counter = 0
    var_groups = OrderedDict()
    dtype = vars[0].dtype

    for var in vars:
        bytes = np.prod(var.shape) * core.size_of_dtype(var.dtype)
        if memory_counter < group_size and dtype == var.dtype:
            memory_counter += bytes
        else:
            memory_counter = bytes
            dtype = var.dtype
            group_idx += 1
        var_groups.setdefault(group_idx, []).append(var)
    return _coalesce_tensors(var_groups)


@imperative_base.no_grad
@framework.dygraph_only
def sync_params_buffers(
    model, comm_group=None, src_rank=0, is_model_parallel=False
):
    model_vars = []
    for _, param in model._obtain_parameters_buffers().items():
        if not isinstance(param, (core.VarBase, core.eager.Tensor)):
            raise TypeError(
                "The data type of '%s' must be Varbase or eager.Tensor"
                % param.name
            )

        # is_distributed param not need to sync when in mp mode
        if isinstance(param, (ParamBase, core.eager.Tensor)):
            if is_model_parallel and param.is_distributed:
                continue

            # NOTE(shenliang03): Support situations that do not require synchronization parameters,
            # such as moe's expert parameters
            if getattr(param, "no_sync", False):
                continue
        if param.type == core.VarDesc.VarType.VOCAB:
            continue

        model_vars.append(param.detach())
    if len(model_vars) == 0:
        return

    # group size is 128M
    coalesced_vars = build_groups(model_vars, 128 * 1024 * 1024)

    for coalesced_var, _, _ in coalesced_vars:
        paddle.distributed.broadcast(
            coalesced_var, src=src_rank, group=comm_group, sync_op=True
        )

    for coalesced_var, origin_vars, var_shapes in coalesced_vars:
        var_len = [np.prod(v_shape) for v_shape in var_shapes]
        paddle.fluid.framework._dygraph_tracer().trace_op(
            type='split',
            inputs={'X': coalesced_var},
            outputs={'Out': origin_vars},
            attrs={'sections': var_len, 'axis': 0},
        )


class DataParallel(layers.Layer):
    """
    Run the dygraph module with data parallelism.

    Currently, DataParallel class only supports to run the dynamic graph
    with multi-process.

    Now supports two ways to start training:

    1. start by ``paddle.distributed.spawn`` method, for example:

        ``python demo.py`` (spawn need to be called in ``__main__`` method)

    2. start by ``paddle.distributed.launch`` module, for example:

        ``python -m paddle.distributed.launch --gpus=0,1 demo.py`` .

    And the content of `demo.py` is the code of examples.

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy, optional): (deprecated) The strategy of data parallelism,
            contains environment configuration related to parallel execution. Default: None.
        comm_buffer_size(int, optional):  It limits the memory size(MB) of one buffer
                                          parameters' gradient which is the input of communication
                                          calling(e.g NCCLAllReduce). Default: 25.
        last_comm_buffer_size(float, optional): It limits memory size(MB) of last buffer in communication
                                         calling. Making the last communication buffer size small is useful to
                                         improve performance. Default: 1.
        find_unused_parameters(bool, optional): Whether to traverse the entire backward graph from the
                                                all tensors in the return value of the wrapped model's
                                                forward function. For parameters not involved in loss
                                                calculation, their gradients will be marked as ready in
                                                advance to prepare reduce. Please note that all forward
                                                outputs derived from the wrapped model parameters must
                                                participate in the calculation of loss and subsequent
                                                gradient calculations. If not, serious error will occur.
                                                Note that setting the find_unused_parameters to True
                                                will affect computing performance. Therefore, if all parameters
                                                are sure to participate in the loss calculation and the
                                                autograd graph construction, please set it False. Default: False.

    Returns:
        Layer: The data paralleled module.

    Examples:

        .. code-block:: python
            :name: dp-example

            # required: distributed
            import paddle
            import paddle.nn as nn
            import paddle.optimizer as opt
            import paddle.distributed as dist

            class LinearNet(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self._linear1 = nn.Linear(10, 10)
                    self._linear2 = nn.Linear(10, 1)

                def forward(self, x):
                    return self._linear2(self._linear1(x))

            def train():
                # 1. initialize parallel environment
                dist.init_parallel_env()

                # 2. create data parallel layer & optimizer
                layer = LinearNet()
                dp_layer = paddle.DataParallel(layer)

                loss_fn = nn.MSELoss()
                adam = opt.Adam(
                    learning_rate=0.001, parameters=dp_layer.parameters())

                # 3. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                loss.backward()

                adam.step()
                adam.clear_grad()

            if __name__ == '__main__':
                # 1. start by ``paddle.distributed.spawn`` (default)
                dist.spawn(train, nprocs=2)
                # 2. start by ``paddle.distributed.launch``
                # train()


    .. note::
        ``PyLayer`` is not supported in DataParallel. To solve problems of this kind,
        it's recommended to skip gradient synchronization among multiple cards by 'no_sync',
        and manually implement 'all_reduce' before model optimization. There is an example
        showing specific implemetation processing.

    Examples:

        .. code-block:: python
            :name: dp-pylayer-example

            # required: distributed
            import numpy
            import paddle
            import paddle.distributed as dist
            from paddle.autograd import PyLayer
            from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

            class cus_tanh(PyLayer):
                @staticmethod
                def forward(ctx, x):
                    y = paddle.tanh(x)
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                def backward(ctx, dy):
                    y, = ctx.saved_tensor()
                    grad = dy * (1 - paddle.square(y))
                    return grad

            class SimpleNet(paddle.nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.linear = paddle.nn.Linear(2, 2)

                def forward(self, inputs):
                    inputs = cus_tanh.apply(inputs)
                    return self.linear(inputs)

            if __name__ == '__main__':
                dist.init_parallel_env()

                model = SimpleNet()
                model = paddle.DataParallel(model)
                opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

                for step in range(10):
                    x_data = numpy.random.randn(2, 2).astype(numpy.float32)
                    x = paddle.to_tensor(x_data)
                    x.stop_gradient = False

                    # step 1 : skip gradient synchronization by 'no_sync'
                    with model.no_sync():
                        y_pred = model(x)
                        loss = y_pred.mean()
                        loss.backward()

                    # step 2 : fuse + allreduce manually before optimization
                    fused_allreduce_gradients(list(model.parameters()), None)

                    opt.step()
                    opt.clear_grad()

    """

    def __init__(
        self,
        layers,
        strategy=None,
        comm_buffer_size=25,
        last_comm_buffer_size=1,
        find_unused_parameters=False,
        group=None,
    ):
        super().__init__(layers.full_name() + "_data_parallel")

        assert (
            in_dygraph_mode()
        ), "It's not supported to construct DataParallel in static graph mode."

        self._layers = layers
        self.find_unused_parameters = find_unused_parameters
        self.grad_need_sync = True
        self.group = group
        self.var_dtype = (
            core.eager.Tensor if in_dygraph_mode() else core.VarBase
        )

        # NOTE(chenweihang): The ParallelStrategy here is not strictly a strategy.
        # It just stores some environment variables, which can be constructed by
        # ParallelEnv. Here it is set as an optional argument.
        # This parameter is not removed because of compatibility with 1.x writing.
        if strategy is not None:
            self._strategy = strategy
        else:
            self._strategy = _build_default_parallel_strategy()

        if self._strategy.nranks > 1:
            # check the environment
            assert parallel_helper.__parallel_ctx__clz__ is not None, (
                "ParallelContext must be initialized before. You should use init_parallel_env() before"
                "constructing the DataParallel."
            )

            if in_dygraph_mode():
                self.group = (
                    paddle.distributed.collective._get_default_group()
                    if self.group is None
                    else self.group
                )

                assert isinstance(
                    self.group, paddle.distributed.collective.Group
                ), "ProcessGroup must be an instance of Group in DataParallel."

            # sync buffer and params
            # TODO(liuyuhui) Currently not support xpu. xpu is
            # still broadcasting parameters when calling layer
            if not paddle.is_compiled_with_xpu():
                sync_params_buffers(self._layers)

            self.comm_buffer_size = int(comm_buffer_size * 1024 * 1024)
            # NOTE(shenliang03): We can set environment variables to control
            # the size of the group, Default: 1MB. The role of this small group is:
            # when the last group allreduce, the overlap cannot work. Making the
            # the last group small is useful to improve performance.
            self.last_comm_buffer_size = int(
                last_comm_buffer_size * 1024 * 1024
            )
            self.init_reducer()
        else:
            warnings.warn(
                "The program will return to single-card operation. "
                "Please check 1, whether you use spawn or fleetrun "
                "to start the program. 2, Whether it is a multi-card "
                "program. 3, Is the current environment multi-card."
            )

    def init_reducer(self):
        layers_param = []
        params_set = set()
        for sublayer in self.sublayers():
            for _, param in sublayer.named_parameters(include_sublayers=False):
                if param is None or param in params_set:
                    continue
                params_set.add(param)
                if not isinstance(param, self.var_dtype):
                    raise TypeError(
                        "The data type of '%s' must be '%s'"
                        % (param.name, self.var_dtype)
                    )
                if param.trainable:
                    layers_param.append((sublayer, param))

        trainable_parameters = list(
            filter(
                lambda x: not getattr(x, "no_sync", False),
                [param for _, param in layers_param],
            )
        )

        assert len(trainable_parameters) > 0, (
            "This model does not have any parameters to train, and "
            "does not need to use DataParallel"
        )

        # NOTE(shenliang03): Here we can only use the attributes to judge whether
        # parameter is sparse(or SelectedRows). The reason is that the sparse message
        # can't be obtained when bp hasn't happened yet. So if layer supports sparse parameter,
        # we should add the layer here like "paddle.nn.layer.common.Embedding".
        def check_layer_sparse(sublayer):
            if isinstance(sublayer, paddle.nn.layer.common.Embedding):
                return sublayer._sparse
            return False

        is_sparse_gradient = [
            check_layer_sparse(sublayer) for sublayer, _ in layers_param
        ]

        if in_dygraph_mode():
            self.group_indices = core.eager_assign_group_by_size(
                trainable_parameters,
                is_sparse_gradient,
                [self.last_comm_buffer_size, self.comm_buffer_size],
            )

            self._reducer = core.EagerReducer(
                trainable_parameters,
                list(reversed(self.group_indices)),
                is_sparse_gradient,
                self.group.process_group,
                [self.last_comm_buffer_size, self.comm_buffer_size],
                self.find_unused_parameters,
            )

    def _find_varbase(self, obj):
        var_type = core.eager.Tensor if in_dygraph_mode() else core.VarBase
        if isinstance(obj, var_type):
            return [obj]
        if isinstance(obj, (list, tuple)):
            return itertools.chain(*map(self._find_varbase, obj))
        if isinstance(obj, dict):
            return itertools.chain(*map(self._find_varbase, obj.values()))
        return []

    @contextmanager
    def no_sync(self):
        """
        A context manager to stop gradient synchronization. Within no_sync(),
        gradients of parameters will only be accumulated on model and not
        synchronized util the first forward-backward out of this context.

        Examples:
            .. code-block:: python

                # required: distributed
                import paddle
                import paddle.nn as nn
                import paddle.distributed as dist

                class SimpleNet(nn.Layer):
                    def __init__(self):
                        super().__init__()
                        self._linear = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear(x)

                dist.init_parallel_env()
                model = SimpleNet()
                dp_model = paddle.DataParallel(model)

                inputs_1 = paddle.randn([10, 10], 'float32')
                inputs_2 = paddle.ones([10, 10], 'float32')

                with dp_model.no_sync():
                    # gradients will not be synchronized
                    dp_model(inputs_1).backward()

                # synchronization happens here
                dp_model(inputs_2).backward()

        """
        tmp_grad_need_sync = self.grad_need_sync
        self.grad_need_sync = False
        try:
            yield
        finally:
            self.grad_need_sync = tmp_grad_need_sync

    def forward(self, *inputs, **kwargs):
        outputs = self._layers(*inputs, **kwargs)
        if (
            self._strategy.nranks > 1
            and framework._dygraph_tracer()._has_grad
            and self.grad_need_sync
        ):
            self._reducer.prepare_for_backward(
                list(self._find_varbase(outputs))
            )
        return outputs

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore."
    )
    def scale_loss(self, loss):
        """
        Deprecated method, now ``scale_loss`` is an empty method,
        keep this method just for compatibility.
        """
        return loss

    @deprecated(
        since="2.0.0", reason="This method does not need to be called anymore."
    )
    def apply_collective_grads(self):
        """
        Deprecated method, now ``apply_collective_grads`` is an empty method,
        keep this method just for compatibility.
        """
        return

    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
    ):
        '''
        Get all parameters and persistable buffers of current layer and its sub-layers. And set them into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters and persistable buffers will be set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters and persistable buffers from sublayers. Default: True

        Retruns:
            dict: a dict contains all the parameters and persistable buffers.

        Examples:
            .. code-block:: python

                import paddle
                import paddle.distributed as dist

                dist.init_parallel_env()

                emb = paddle.nn.Embedding(10, 10)
                emb = paddle.fluid.dygraph.DataParallel(emb)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

        '''

        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
        )

    @framework.deprecate_stat_dict
    def set_state_dict(self, state_dict, use_structured_name=True):
        '''
        Set parameters and persistable buffers from state_dict. All the parameters and buffers will be reset by the tensor in the state_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters and persistable buffers.
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter or buffer name as key.
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle
                import paddle.distributed as dist

                dist.init_parallel_env()

                emb = paddle.nn.Embedding(10, 10)
                emb = paddle.fluid.dygraph.DataParallel(emb)

                state_dict = emb.state_dict()
                paddle.save(state_dict, "paddle_dy.pdparams")

                para_state_dict = paddle.load("paddle_dy.pdparams")
                emb.set_state_dict(para_state_dict)

        '''

        self._layers.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )

    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict

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
import six
import numpy as np
from collections import OrderedDict
from .. import core
from . import layers
from . import parallel_helper
from .. import framework
from ..layers import collective
from . import to_variable, no_grad

__all__ = ["prepare_context", "ParallelEnv", "DataParallel"]

ParallelStrategy = core.ParallelStrategy


def prepare_context(strategy=None):
    '''
    :api_attr: imperative
    '''
    if strategy is None:
        strategy = ParallelStrategy()
        strategy.nranks = Env().nranks
        strategy.local_rank = Env().local_rank
        strategy.trainer_endpoints = Env().trainer_endpoints
        strategy.current_endpoint = Env().current_endpoint
    if strategy.nranks < 2:
        return
    assert framework.in_dygraph_mode() is True, \
        "dygraph.prepare_context should be used with dygrahp mode."
    place = framework._current_expected_place()
    assert place is not None, \
        "dygraph.prepare_context should be used in fluid.dygraph.guard(place) guard."
    if isinstance(place, core.CUDAPlace):
        parallel_helper._set_parallel_ctx(
            core.NCCLParallelContext(strategy, place))
    else:
        # TODO(Yancey1989): add Gloo Parallel Context to support CPU parallel computation
        assert ("Only support CUDAPlace for now.")
    parallel_helper._init_parallel_ctx()
    return strategy


class ParallelEnv(object):
    """
    **Notes**:
        **The old class name was Env and will be deprecated. Please use new class name ParallelEnv.**

    This class is used to obtain the environment variables required for 
    the parallel execution of dynamic graph model.

    The dynamic graph parallel mode needs to be started using paddle.distributed.launch.
    By default, the related environment variable is automatically configured by this module.

    This class is generally used in with `fluid.dygraph.DataParallel` to configure dynamic graph models
    to run in parallel.

    Examples:
      .. code-block:: python

        # This example needs to run with paddle.distributed.launch, The usage is:
        #   python -m paddle.distributed.launch --selected_gpus=0,1 example.py
        # And the content of `example.py` is the code of following example.

        import numpy as np
        import paddle.fluid as fluid
        import paddle.fluid.dygraph as dygraph
        from paddle.fluid.optimizer import AdamOptimizer
        from paddle.fluid.dygraph.nn import Linear
        from paddle.fluid.dygraph.base import to_variable

        place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
        with fluid.dygraph.guard(place=place):

            # prepare the data parallel context
            strategy=dygraph.prepare_context()

            linear = Linear(1, 10, act="softmax")
            adam = fluid.optimizer.AdamOptimizer()

            # make the module become the data parallelism module
            linear = dygraph.DataParallel(linear, strategy)

            x_data = np.random.random(size=[10, 1]).astype(np.float32)
            data = to_variable(x_data)

            hidden = linear(data)
            avg_loss = fluid.layers.mean(hidden)

            # scale the loss according to the number of trainers.
            avg_loss = linear.scale_loss(avg_loss)

            avg_loss.backward()

            # collect the gradients of trainers.
            linear.apply_collective_grads()

            adam.minimize(avg_loss)
            linear.clear_gradients()
    """

    def __init__(self):
        self._nranks = int(os.getenv("PADDLE_TRAINERS_NUM", "1"))
        self._local_rank = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        self._dev_id = int(os.getenv("FLAGS_selected_gpus", "0"))
        self._trainer_endpoints = os.getenv("PADDLE_TRAINER_ENDPOINTS",
                                            "").split(",")
        self._current_endpoint = os.getenv("PADDLE_CURRENT_ENDPOINT", "")

    @property
    def nranks(self):
        """
        The number of trainers, generally refers to the number of GPU cards used in training.

        Its value is equal to the value of the environment variable PADDLE_TRAINERS_NUM. The default value is 1.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINERS_NUM=4
            import paddle.fluid as fluid
            
            env = fluid.dygraph.ParallelEnv()
            print("The nranks is %d" % env.nranks)
            # The nranks is 4
        """
        return self._nranks

    @property
    def local_rank(self):
        """
        The current trainer number.

        Its value is equal to the value of the environment variable PADDLE_TRAINER_ID. The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ID=0
            import paddle.fluid as fluid
            
            env = fluid.dygraph.ParallelEnv()
            print("The local rank is %d" % env.local_rank)
            # The local rank is 0
        """
        return self._local_rank

    @property
    def dev_id(self):
        """
        The ID of selected GPU card for parallel training.

        Its value is equal to the value of the environment variable FLAGS_selected_gpus. The default value is 0.

        Examples:
          .. code-block:: python

            # execute this command in terminal: export FLAGS_selected_gpus=1
            import paddle.fluid as fluid
            
            env = fluid.dygraph.ParallelEnv()
            print("The device id are %d" % env.dev_id)
            # The device id are 1
        """
        return self._dev_id

    @property
    def current_endpoint(self):
        """
        The endpoint of current trainer, it is in the form of (node IP + port).

        Its value is equal to the value of the environment variable PADDLE_CURRENT_ENDPOINT. The default value is "".

        Examples:
          .. code-block:: python
            
            # execute this command in terminal: export PADDLE_CURRENT_ENDPOINT=127.0.0.1:6170
            import paddle.fluid as fluid
            
            env = fluid.dygraph.ParallelEnv()
            print("The current endpoint are %s" % env.current_endpoint)
            # The current endpoint are 127.0.0.1:6170
        """
        return self._current_endpoint

    @property
    def trainer_endpoints(self):
        """
        The endpoints of all trainer nodes in the task, 
        which are used to broadcast the NCCL ID when NCCL2 is initialized.

        Its value is equal to the value of the environment variable PADDLE_TRAINER_ENDPOINTS. The default value is "".

        Examples:
          .. code-block:: python

            # execute this command in terminal: export PADDLE_TRAINER_ENDPOINTS=127.0.0.1:6170,127.0.0.1:6171
            import paddle.fluid as fluid
            
            env = fluid.dygraph.ParallelEnv()
            print("The trainer endpoints are %s" % env.trainer_endpoints)
            # The trainer endpoints are ['127.0.0.1:6170', '127.0.0.1:6171']
        """
        return self._trainer_endpoints


# NOTE: [ Compatible ] Originally this class name is `Env`. The semantics of the old class names
# are inaccurate and may confuse users, so replace it with `ParallelEnv`, but to be compatible
# with the old examples, here still need to keep this name.
Env = ParallelEnv


class DataParallel(layers.Layer):
    """
    Run the dygraph module with data parallelism.

    Currently, DataParallel class only supports to run the dynamic graph
    with multi-process. The usage is:
    `python -m paddle.distributed.launch --selected_gpus=0,1 dynamic_graph_test.py`.
    And the content of `dynamic_graph_test.py` is the code of examples.

    Args:
        layers(Layer): The module that should be executed by data parallel.
        strategy(ParallelStrategy): The strategy of data parallelism, contains 
            environment configuration related to parallel execution.

    Returns:
        Layer: The data paralleled module.

    Examples:
        .. code-block:: python

           import numpy as np
           import paddle.fluid as fluid
           import paddle.fluid.dygraph as dygraph
           from paddle.fluid.optimizer import AdamOptimizer
           from paddle.fluid.dygraph.nn import Linear
           from paddle.fluid.dygraph.base import to_variable

           place = place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
           with fluid.dygraph.guard(place=place):

               # prepare the data parallel context
               strategy=dygraph.prepare_context()

               linear = Linear(1, 10, act="softmax")
               adam = fluid.optimizer.AdamOptimizer()

               # make the module become the data parallelism module
               linear = dygraph.DataParallel(linear, strategy)

               x_data = np.random.random(size=[10, 1]).astype(np.float32)
               data = to_variable(x_data)

               hidden = linear(data)
               avg_loss = fluid.layers.mean(hidden)

               # scale the loss according to the number of trainers.
               avg_loss = linear.scale_loss(avg_loss)

               avg_loss.backward()

               # collect the gradients of trainers.
               linear.apply_collective_grads()

               adam.minimize(avg_loss)
               linear.clear_gradients()
    """

    def __init__(self, layers, strategy):
        super(DataParallel,
              self).__init__(layers.full_name() + "_data_parallel")

        self._layers = layers
        self._strategy = strategy

    def forward(self, *inputs, **kwargs):
        return self._layers(*inputs, **kwargs)

    def scale_loss(self, loss):
        """
        Scale the loss. In data parallel mode, the loss should be scale with
        the number of trainers. If not in data parallel mode, return the loss
        directly.

        Args:
            loss(Variable): The loss of the current Model.

        Returns:
            Variable: the scaled loss.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                import paddle.fluid.dygraph as dygraph
                from paddle.fluid.optimizer import AdamOptimizer
                from paddle.fluid.dygraph.nn import Linear
                from paddle.fluid.dygraph.base import to_variable

                place = place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
                with fluid.dygraph.guard(place=place):
                    strategy=dygraph.prepare_context()
                    linear = Linear(1, 10, act="softmax")
                    adam = fluid.optimizer.AdamOptimizer()
                    linear = dygraph.DataParallel(linear, strategy)

                    x_data = np.random.random(size=[10, 1]).astype(np.float32)
                    data = to_variable(x_data)
                    hidden = linear(data)
                    avg_loss = fluid.layers.mean(hidden)

                    # scale the loss according to the number of trainers.
                    avg_loss = linear.scale_loss(avg_loss)

                    avg_loss.backward()
                    linear.apply_collective_grads()

                    adam.minimize(avg_loss)
                    linear.clear_gradients()
        """
        if not self._is_data_parallel_mode():
            return loss

        loss_scale = to_variable(
            np.array([self._strategy.nranks]).astype("float32"))
        loss_scale.stop_gradient = True
        loss = loss / loss_scale
        return loss

    def _coalesce_tensors(self, var_groups):
        from ..layers import nn
        coalesced_grads_and_grad_vars = []
        for group_id, grad_vars in var_groups.items():
            flattened_vars = []
            g_var_shapes = []
            for g_var in grad_vars:
                g_var_shapes.append(g_var.shape)
                flattened_vars.append(
                    nn.reshape(
                        x=g_var, shape=[np.prod(g_var.shape)], inplace=True))
            coalesced_grad = nn.concat(flattened_vars)
            coalesced_grads_and_grad_vars.append(
                [coalesced_grad, grad_vars, g_var_shapes])
        return coalesced_grads_and_grad_vars

    def _reshape_inplace(self, x, shape):
        x_shape = self._helper.create_variable_for_type_inference(dtype=x.dtype)
        self._helper.append_op(
            type="reshape2",
            inputs={'X': x},
            attrs={'shape': shape},
            outputs={'Out': x,
                     'XShape': x_shape})

    def _split_tensors(self, coalesced_grads_and_grad_vars):
        from ..layers import nn
        for coalesced_grad, origin_grad_vars, grad_shapes in coalesced_grads_and_grad_vars:
            grad_var_len = [np.prod(g_shape) for g_shape in grad_shapes]
            self._helper.main_program.current_block().append_op(
                type='split',
                inputs={'X': coalesced_grad},
                outputs={'Out': origin_grad_vars},
                attrs={'sections': grad_var_len,
                       'axis': 0})
            for g_var, g_shape in zip(origin_grad_vars, grad_shapes):
                self._reshape_inplace(x=g_var, shape=g_shape)
                assert g_var.shape == g_shape

    @no_grad
    def apply_collective_grads(self):
        """
        AllReduce the Parameters' gradient.

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid
                import paddle.fluid.dygraph as dygraph
                from paddle.fluid.optimizer import AdamOptimizer
                from paddle.fluid.dygraph.nn import Linear
                from paddle.fluid.dygraph.base import to_variable

                place = place = fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
                with fluid.dygraph.guard(place=place):
                    strategy=dygraph.prepare_context()
                    linear = Linear(1, 10, act="softmax")
                    adam = fluid.optimizer.AdamOptimizer()
                    linear = dygraph.DataParallel(linear, strategy)

                    x_data = np.random.random(size=[10, 1]).astype(np.float32)
                    data = to_variable(x_data)
                    hidden = linear(data)
                    avg_loss = fluid.layers.mean(hidden)
                    avg_loss = linear.scale_loss(avg_loss)
                    avg_loss.backward()

                    # collect the gradients of trainers.
                    linear.apply_collective_grads()

                    adam.minimize(avg_loss)
                    linear.clear_gradients()
        """
        if not self._is_data_parallel_mode():
            return

        grad_var_set = set()
        grad_vars = []
        for param in self._layers.parameters():
            # NOTE(zcd): The grad_ivar maybe no generated.
            if param.trainable and (param._grad_ivar() is not None):
                g_var = param._grad_ivar()
                grad_vars.append(g_var)
                assert g_var not in grad_var_set
                grad_var_set.add(g_var)

        # FIXME(zcd): the type of the var should be LoDTensor, i.e
        # the gradients should be dense, otherwise, the following
        # logic should be updated.
        # 128 MB as a group
        mega_bytes = 128 * 1024 * 1024
        group_idx = 0
        memory_counter = 0
        grad_var_groups = OrderedDict()
        dtype = grad_vars[0].dtype
        for g_var in grad_vars:
            # Note: the dtype of the same group should be the same.
            bytes = np.prod(g_var.shape) * core.size_of_dtype(g_var.dtype)
            if memory_counter < mega_bytes and dtype == g_var.dtype:
                memory_counter += bytes
            else:
                memory_counter = bytes
                group_idx += 1
            grad_var_groups.setdefault(group_idx, []).append(g_var)

        coalesced_grads_and_vars = self._coalesce_tensors(grad_var_groups)

        for coalesced_grad, g_vars, g_shapes in coalesced_grads_and_vars:
            collective._allreduce(
                coalesced_grad, coalesced_grad, sync_mode=False)

        self._split_tensors(coalesced_grads_and_vars)

    def _is_data_parallel_mode(self):
        return self._strategy.nranks > 1

    def state_dict(self,
                   destination=None,
                   include_sublayers=True,
                   structured_name_prefix=""):
        '''
        Get all parameters of self._layers and its sub-layers. And set all the parameters into a dict

        Parameters:
            destination(dict, optional) : If provide, all the parameters will set to this dict . Default: None
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            structured_name_prefix(str, optional): If not empty str, all the key in state dict will start 
                                                 with structured_name_prefix

        Retruns:
            dict: a dict contains all the parameters of self._layers

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    strategy=fluid.dygraph.prepare_context()
                    emb = fluid.dygraph.Embedding([10, 10])
                    emb = fluid.dygraph.DataParallel(emb, strategy)

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")

        '''

        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix)

    def set_dict(self,
                 stat_dict,
                 include_sublayers=True,
                 use_structured_name=True):
        '''
        Set parameters of self._layers from stat_dict. All the parameters of self._layers will be reset by the tensor in the stat_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter name as key. 
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    strategy=fluid.dygraph.prepare_context()
                    emb = fluid.dygraph.Embedding([10, 10])
                    emb = fluid.dygraph.DataParallel(emb, strategy)

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")
                    
                    para_state_dict, _ = fluid.load_dygraph( "paddle_dy")

                    emb.set_dict( para_state_dict )

        '''

        self._layers.set_dict(
            stat_dict,
            include_sublayers=include_sublayers,
            use_structured_name=use_structured_name)

    def load_dict(self,
                  stat_dict,
                  include_sublayers=True,
                  use_structured_name=True):
        '''
        Set parameters of self._layers from stat_dict. All the parameters of self._layers will be reset by the tensor in the stat_dict

        This api will be Deprecated. Please use set_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name as key, otherwise, use parameter name as key.
                                                  Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    strategy=fluid.dygraph.prepare_context()
                    emb = fluid.dygraph.Embedding([10, 10])
                    emb = fluid.dygraph.DataParallel(emb, strategy)

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")
                    
                    para_state_dict, _ = fluid.load_dygraph( "paddle_dy")

                    emb.load_dict( para_state_dict )

        '''

        self._layers.load_dict(
            stat_dict,
            include_sublayers=include_sublayers,
            use_structured_name=use_structured_name)

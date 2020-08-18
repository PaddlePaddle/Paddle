# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
from paddle.fluid.framework import in_dygraph_mode

from .device import _get_device


def monkey_patch_layer():
    def load_dict(self,
                  stat_dict,
                  include_sublayers=True,
                  use_structured_name=True):
        '''
        Set parameters from stat_dict. All the parameters will be reset by the
        tensor in the stat_dict

        This api will be Deprecated. Please use set_dict

        Parameters:
            state_dict(dict) : Dict contains all the parameters
            include_sublayers(bool, optional) : If true, also include the
                parameters from sublayers. Default: True
            use_structured_name(bool, optional) : If true, use structured name
                as key, otherwise, use parameter name as key. Default: True
        Returns:
            None

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                with fluid.dygraph.guard():
                    emb = fluid.dygraph.Embedding([10, 10])

                    state_dict = emb.state_dict()
                    fluid.save_dygraph( state_dict, "paddle_dy")
                    
                    para_state_dict, _ = fluid.load_dygraph( "paddle_dy")
                    emb.load_dict( para_state_dict )

        '''

        def _check_match(key, param):
            state = stat_dict.get(key, None)
            if state is None:
                raise ValueError(
                    "{} is not found in the providing file.".format(key))
            if list(state.shape) != list(param.shape):
                raise ValueError(
                    "{} receives a shape {}, but the expected shape is {}.".
                    format(key, list(state.shape), list(param.shape)))
            return param, state

        matched_param_state = []
        for key, param in self.state_dict().items():
            key_name = key if use_structured_name else param.name
            try:
                match_res = _check_match(key_name, param)
                matched_param_state.append(match_res)
            except ValueError as err:
                warnings.warn(("Skip loading for {}. ".format(key) + str(err)))

        if in_dygraph_mode():
            for param, state in matched_param_state:
                param.set_value(state)
        else:

            def _set_var(var, ndarray):
                t = fluid.global_scope().find_var(var.name).get_tensor()
                p = t._place()
                if p.is_cpu_place():
                    place = fluid.CPUPlace()
                elif p.is_cuda_pinned_place():
                    place = fluid.CUDAPinnedPlace()
                else:
                    p = fluid.core.Place()
                    p.set_place(t._place())
                    place = fluid.CUDAPlace(p.gpu_device_id())
                t.set(ndarray, place)

            executor = fluid.Executor(_get_device())._default_executor
            # restore parameter states
            fluid.core._create_loaded_parameter(
                [param for param, state in matched_param_state],
                fluid.global_scope(), executor)
            for param, state in matched_param_state:
                _set_var(param, state)

    setattr(fluid.dygraph.Layer, 'load_dict', load_dict)

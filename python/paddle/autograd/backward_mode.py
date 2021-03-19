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

from paddle.fluid import core
from paddle.fluid import framework
import paddle
__all__ = ['backward']


@framework.dygraph_only
def backward(tensors,
             grad_tensors,
             retain_graph=None,
             create_graph=False,
             inputs=None):
    def check_tensors(in_out_list, name):
        assert in_out_list is not None, "{} should not be None".format(name)

        if isinstance(in_out_list, (list, tuple)):
            assert len(in_out_list) > 0, "{} connot be empyt".format(name)
            for each_var in in_out_list:
                assert isinstance(
                    each_var, paddle.
                    Tensor), "Elements of {} must be paddle.Tensor".format(name)
            return in_out_list
        else:
            assert isinstance(
                in_out_list,
                paddle.Tensor), "{} must be Tensor or list of Tensor".format(
                    name)
            return [in_out_list]

    tensors = check_tensors(tensors, "tensors")

    if grad_tensors is not None:
        if not isinstance(grad_tensors, (list, tuple)):
            grad_tensors = [grad_tensors]

        for each_tensor in grad_tensors:
            if each_tensor is not None:
                assert isinstance(
                    each_tensor, paddle.Tensor
                ), "grad_tensors must be None, Tensor or list containing None or Tensor"
    else:
        grad_tensors = []

    if len(grad_tensors) > 0:
        assert len(tensors) == len(
            grad_tensors), "The length of grad_tensors must be equal to tensors"

    assert isinstance(create_graph, bool), "create_graph must be True or False"

    if retain_graph is None:
        retain_graph = create_graph

    assert isinstance(retain_graph,
                      bool), "retain_graph must be None, True or False"

    if inputs is not None:
        assert len(inputs) > 0, "inputs cannot be empty list"

    core.dygraph_run_backward(tensors, grad_tensors, retain_graph, create_graph,
                              inputs, framework._dygraph_tracer())

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

from paddle.utils import try_import

__all__ = ['export']


def export(layer, save_file, input_spec=None, opset_version=9, **configs):
    """
    Export Layer as ONNX format model, which can be used for inference.
    Now, it supports a limited operater set and dynamic models.(e.g., MobileNet.)
    More features and introduction, Please reference the https://github.com/PaddlePaddle/paddle2onnx
    
    Args:
        layer (Layer): The Layer to be saved.
        save_file (str): The file path to save the onnx model.
        input_spec (list[InputSpec|Tensor], optional): Describes the input of the saved model. 
            It is the example inputs that will be passed to saved ONNX model.
            If None, Please specific `input_spec` to layer in `@paddle.jit.to_static`. Default None.
        opset_version(int, optional): Opset version of exported ONNX model.
            Now, stable supported opset version include 9, 10, 11. Default 9.
        **configs (dict, optional): Additional keyword parameters. Currently supports 'output_spec', 
            which describes the output to prune model.
    Returns:
        None
    Examples:
        .. code-block:: python
			import paddle
			import numpy as np
			
			class LinearNet(paddle.nn.Layer):
			    def __init__(self):
			        super(LinearNet, self).__init__()
			        self._linear = paddle.nn.Linear(128, 10)
			
			    def forward(self, x):
			        return self._linear(x)
			
			#export model with InputSpec, which supports set dynamic shape for inputs.
			def export_linear_net():
			    model = LinearNet()
			    x_spec = paddle.static.InputSpec(shape=[None, 128], dtype='float32')
			    paddle.onnx.export(model, 'linear_net.onnx', input_spec=[x_spec])
			
			export_linear_net()
			
			
			class Logic(paddle.nn.Layer):
			    def __init__(self):
			        super(Logic, self).__init__()
			
			    def forward(self, x, y, z):
			        if z:
			            return x
			        else:
			            return y
			
			#export model with Tensor, which supports prune model by set 'output_spec' with output of model.
			def export_logic():
			    model = Logic()
			    x = paddle.to_tensor(np.random.random((1)))
			    y = paddle.to_tensor(np.random.random((1)))
			    # prune model with input_spec and output_spec, which need to static and run model before export.
			    paddle.jit.to_static(model)
			    out = model(x, y, z=True)
			    paddle.onnx.export(model, 'pruned.onnx', input_spec=[x], output_spec=[out])
			
			export_logic()
    """

    p2o = try_import('paddle2onnx')

    p2o.dygraph2onnx(
        layer,
        save_file,
        input_spec=input_spec,
        opset_version=opset_version,
        **configs)

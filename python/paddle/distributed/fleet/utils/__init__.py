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

from .fs import LocalFS  # noqa: F401
from .fs import HDFSClient  # noqa: F401
from .ps_util import DistributedInfer  # noqa: F401
import paddle.utils.deprecated as deprecated
from paddle.distributed import fleet

import paddle
from . import log_util  # noqa: F401
from . import hybrid_parallel_util  # noqa: F401

__all__ = ["LocalFS", "recompute", "DistributedInfer", "HDFSClient"]  # noqa


def recompute(function, *args, **kwargs):
    """
    recompute intermediate activations to save then memory.
    Parameters:
        function(paddle.nn.Layer): layer of sequence of layers that describes part of forward pass of the model
              whose intermediate activations will be released to save memory in forward stage and will be recomputed
              in backward stage for gradient calculation.
        *args(Tensor): inputs to the function.
        **kwargs(Dict): Kwargs should only contain the key-value pair of preserve_rng_state, which is used to
              indicate whether to save the forward rng. If it is True, then the last forward rng value will be
              restored when the forward recalculation of backpropagation is performed. The default
              preserve_rng_state is True.
    Returns:
        Output of function on args.

    Examples:
        .. code-block:: python

            import paddle
            from paddle.distributed.fleet.utils import recompute
            import random
            # required: gpu
            def get_fc_block(block_idx, input_size, is_last=False):
                block_name = "block_" + str(block_idx)
                block = paddle.nn.Sequential(
                    (block_name + "_fc_0", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
                    (block_name + "_dropout", paddle.nn.Dropout(p=0.5)),
                    (block_name + "_relu_1", paddle.nn.ReLU()),
                    (block_name + "_fc_1", paddle.nn.Linear(input_size, input_size, bias_attr=False)),
                    (block_name + "_relu_2", paddle.nn.ReLU()),
                )
                if is_last:
                    block.add_sublayer(
                        block_name + "_fc_2",
                        paddle.nn.Linear(
                            input_size, 1, bias_attr=False
                        )
                    )
                else:
                    block.add_sublayer(
                        block_name + "_fc_2",
                        paddle.nn.Linear(input_size, input_size, bias_attr=False)
                    )
                return block
            class Naive_fc_net(paddle.nn.Layer):
                def __init__(self, input_size=10,
                            recompute_blocks=[1, 3],
                            recompute_kwargs={}):
                    super(Naive_fc_net, self).__init__()
                    self.recompute_blocks = recompute_blocks
                    self.recompute_kwargs = recompute_kwargs
                    self.runfunc0 = get_fc_block(0, input_size, is_last=False)
                    self.runfunc1 = get_fc_block(1, input_size, is_last=False)
                    self.runfunc2 = get_fc_block(2, input_size, is_last=False)
                    self.runfunc3 = get_fc_block(3, input_size, is_last=False)
                    self.runfunc4 = get_fc_block(4, input_size, is_last=True)
                    self.total_func = [self.runfunc0, self.runfunc1, self.runfunc2, self.runfunc3, self.runfunc4]
                def forward(self, inputs):
                    nums = len(self.total_func)
                    for i in range(nums):
                        if i in self.recompute_blocks:
                            inputs = recompute(self.total_func[i], inputs, **{"preserve_rng_state": True})
                        else:
                            inputs = self.total_func[i](inputs)
                    return inputs
            def run_model(cuda_state, recompute_block=[], recompute_kwargs={}):
                gen = paddle.seed(10)
                gen.manual_seed(10)
                random.seed(10)
                if cuda_state:
                    paddle.set_cuda_rng_state(cuda_state)
                batch_size, input_size = 1, 10
                model = Naive_fc_net(
                    input_size,
                    recompute_blocks=recompute_block,
                    recompute_kwargs=recompute_kwargs)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                loss_ = []
                param_ = []
                grad_ = []
                for _ in range(5):
                    x = paddle.rand(shape=[batch_size, input_size], dtype="float32")
                    y_pred = model(x)
                    loss = y_pred.mean()
                    loss_.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    param_.append(model.parameters()[9])
                    grad_.append(model.parameters()[3]._grad_ivar())
                    optimizer.clear_grad()
                return loss_, param_, grad_
            cuda_state = paddle.get_cuda_rng_state()
            # without recompute
            loss_ref, param_ref, grad_ref = run_model(
                cuda_state, recompute_block=[]
            )
            loss, param, grad = run_model(cuda_state, recompute_block=[1, 2])
            print("normal_loss: {}, recompute_loss: {}".format(loss_ref, loss))
            # The result of the recompute_loss should be the same as the normal_loss.
    """

    return fleet.recompute.recompute(function, *args, **kwargs)

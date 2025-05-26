# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License


import paddle
import paddle.distributed as dist

_enable_auto_dp_mode = False


def _fake_replicate_grad_to_partial(grad, partial_axis):
    new_placements = grad.placements
    assert (
        new_placements[partial_axis] == dist.Replicate()
    ), "when reshard fake replicated grad to partial, the partial axis of grad should be Replicate"

    new_placements[partial_axis] = dist.Partial(dist.ReduceType.kRedSum)

    grad_mesh = grad.process_mesh
    grad = dist.auto_parallel.api.dtensor_to_local(
        grad, grad_mesh, grad.placements
    )
    grad = dist.auto_parallel.api.dtensor_from_local(
        grad, grad_mesh, new_placements
    )
    return grad


def _convert_fake_replicate_grad_to_partial(params_grads):
    # skip non-parallel cases
    word_size = paddle.distributed.get_world_size()
    if word_size == 1:
        return

    if isinstance(params_grads, list):
        for idx in range(len(params_grads)):
            param, grad = params_grads[idx][0], params_grads[idx][1]
            if grad.is_dist():
                grad_placements = grad.placements
                if not isinstance(grad_placements[0], dist.Partial):
                    grad = _fake_replicate_grad_to_partial(grad, 0)
            else:
                default_grad_placements = [
                    dist.Partial(dist.ReduceType.kRedSum)
                ]
                default_grad_mesh = dist.ProcessMesh(
                    list(range(0, word_size)), dim_names=["dp"]
                )
                grad = dist.auto_parallel.api.dtensor_from_local(
                    grad, default_grad_mesh, default_grad_placements
                )
            params_grads[idx] = (param, grad)
    else:
        for idx in range(len(params_grads['params'])):
            grad = params_grads['params'][idx][1]
            if grad.is_dist():
                grad_placements = grad.placements
                if not isinstance(grad_placements[0], dist.Partial):
                    grad = _fake_replicate_grad_to_partial(grad, 0)
            else:
                default_grad_placements = [
                    dist.Partial(dist.ReduceType.kRedSum)
                ]
                default_grad_mesh = dist.ProcessMesh(
                    list(range(0, word_size)), dim_names=["dp"]
                )
                grad = dist.auto_parallel.api.dtensor_from_local(
                    grad, default_grad_mesh, default_grad_placements
                )
            params_grads['params'][idx] = (params_grads['params'][idx][0], grad)


def in_auto_dp_mode():
    word_size = paddle.distributed.get_world_size()
    if word_size <= 1:
        return False

    global _enable_auto_dp_mode
    return _enable_auto_dp_mode


def _enable_auto_dp():
    global _enable_auto_dp_mode
    _enable_auto_dp_mode = True

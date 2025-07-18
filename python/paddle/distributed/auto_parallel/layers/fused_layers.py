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
# limitations under the License.
import os

import paddle
import paddle.distributed as dist
from paddle import _C_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger
from paddle.framework import core

_raise_cuda_env_unset_warning = True
_mp_async_allreduce = False
_sp_async_reduce_scatter = False


def check_environment_for_overlap():
    if int(os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", "0")) != 1:
        global _raise_cuda_env_unset_warning
        if _raise_cuda_env_unset_warning:
            logger.warning(
                "You set enable_mp_async_allreduce or enable_sp_async_reduce_scatter, but you forget to set environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS=1, which may leads to performance "
                "loss. Try to export CUDA_DEVICE_MAX_CONNECTIONS=1 for better performance."
            )
            _raise_cuda_env_unset_warning = False


def is_fused_matmul_bias_supported():
    if (
        paddle.is_compiled_with_cuda()
        and not paddle.is_compiled_with_rocm()
        or paddle.is_compiled_with_xpu()
    ):
        return hasattr(core.eager.ops.legacy, "fused_gemm_epilogue")
    else:
        return False


if is_fused_matmul_bias_supported():
    origin_linear = paddle.incubate.nn.functional.fused_linear
else:
    origin_linear = paddle.nn.functional.linear


def mp_async_allreduce(x_grad):
    if _mp_async_allreduce and x_grad.process_mesh is not None:
        check_environment_for_overlap()
        mp_placement_index = x_grad.process_mesh.dim_names.index("mp")
        if (
            mp_placement_index != -1
            and x_grad.placements[mp_placement_index].is_partial()
        ):
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            task = dist.stream.all_reduce(
                x_grad._local_value(),
                group=model_parallel_group,
                sync_op=False,
            )
            return task
        else:
            return None
    else:
        return None


def sp_async_reducesctter(x_grad):
    if _sp_async_reduce_scatter and x_grad.process_mesh is not None:
        check_environment_for_overlap()
        mp_placement_index = x_grad.process_mesh.dim_names.index("mp")
        if (
            mp_placement_index != -1
            and x_grad.placements[mp_placement_index].is_partial()
        ):
            hcg = fleet.get_hybrid_communicate_group()
            model_parallel_group = hcg.get_model_parallel_group()
            parallelism = model_parallel_group.nranks

            assert (
                x_grad.shape[0] % parallelism == 0
            ), f"Input sequence length {x_grad.shape[0]} can't be divided exactly by sequence parallelism {parallelism}"

            # reduce-scatter dx
            x_grad_global_shape = x_grad.shape
            x_grad_global_shape[0] = x_grad_global_shape[0] // parallelism
            x_grad_local = x_grad._local_value()
            x_grad_local_shape = x_grad_local.shape
            x_grad_local_shape[0] = x_grad_local_shape[0] // parallelism
            dx_local = paddle.empty(
                shape=x_grad_local_shape, dtype=x_grad.dtype
            )
            task = dist.stream.reduce_scatter(
                dx_local,
                x_grad_local,
                op=dist.ReduceOp.SUM,
                group=model_parallel_group,
                sync_op=False,
            )
            return task, dx_local, x_grad_global_shape
        else:
            return None
    else:
        return None


def sync_mp_allreduce(task, dist_tensor):
    mp_placement_index = dist_tensor.process_mesh.dim_names.index("mp")
    new_placements = []
    for idx, placement in enumerate(dist_tensor.placements):
        if idx == mp_placement_index:
            new_placements.append(dist.Replicate())
        else:
            new_placements.append(placement)
    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    task.wait()

    return paddle.Tensor(
        dist_tensor._local_value(),
        dims=dist_tensor.shape,
        process_mesh=dist_tensor.process_mesh,
        placements=new_placements,
        place=place,
    )


def sync_sp_reducescatter(task, dist_tensor):
    task, dx_local, x_grad_global_shape = task
    placements = [dist.Shard(1), dist.Shard(0)]
    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    task.wait()

    return paddle.Tensor(
        dx_local,
        dims=x_grad_global_shape,
        process_mesh=dist_tensor.process_mesh,
        placements=placements,
        place=place,
    )


# modify from Paddle/python/paddle/distributed/auto_parallel/moe_utils.py
def dist_reshape(dist_tensor):
    local_tensor = dist_tensor._local_value()
    tgt_global_shape = [
        dist_tensor.shape[0] * dist_tensor.shape[1],
        dist_tensor.shape[2],
    ]
    tgt_local_shape = [
        local_tensor.shape[0] * local_tensor.shape[1],
        local_tensor.shape[2],
    ]

    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    local_tensor = local_tensor.reshape(tgt_local_shape)

    if dist_tensor.placements[1].is_shard():
        new_placements = [dist.Shard(0), dist.Shard(1)]
    else:
        new_placements = [dist.Shard(0), dist.Replicate()]

    out = paddle.Tensor(
        local_tensor,
        dims=tgt_global_shape,
        process_mesh=dist_tensor.process_mesh,
        placements=new_placements,
        place=place,
    )
    out.stop_gradient = dist_tensor.stop_gradient
    return out


class FusedLinearWithGradAdd(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        if _sp_async_reduce_scatter:
            task = sp_async_reducesctter(x_grad)
        else:
            task = mp_async_allreduce(x_grad)

        # _C_ops.fused_linear_param_grad_add(x, y_grad, weight_grad, bias_grad, multi precision, has bias)
        if bias is None:
            if hasattr(weight, "main_grad"):
                weight.main_grad, _ = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.main_grad, None, True, False
                )
                if task is not None:
                    if _sp_async_reduce_scatter:
                        x_grad = sync_sp_reducescatter(task, x_grad)
                    else:
                        x_grad = sync_mp_allreduce(task, x_grad)
                return x_grad, None
            else:
                if weight.grad is not None:
                    weight.grad, _ = _C_ops.fused_linear_param_grad_add(
                        x,
                        y_grad,
                        weight.grad,
                        None,
                        False if weight.grad.dtype != paddle.float32 else True,
                        False,
                    )
                    if task is not None:
                        if _sp_async_reduce_scatter:
                            x_grad = sync_sp_reducescatter(task, x_grad)
                        else:
                            x_grad = sync_mp_allreduce(task, x_grad)
                    return x_grad, None
                else:
                    weight_grad, _ = _C_ops.fused_linear_param_grad_add(
                        x, y_grad, None, None, False, False
                    )
                    if task is not None:
                        if _sp_async_reduce_scatter:
                            x_grad = sync_sp_reducescatter(task, x_grad)
                        else:
                            x_grad = sync_mp_allreduce(task, x_grad)
                    return x_grad, weight_grad

        if hasattr(weight, "main_grad") and hasattr(bias, "main_grad"):
            weight.main_grad, bias.main_grad = (
                _C_ops.fused_linear_param_grad_add(
                    x, y_grad, weight.main_grad, bias.main_grad, True, True
                )
            )
            if task is not None:
                if _sp_async_reduce_scatter:
                    x_grad = sync_sp_reducescatter(task, x_grad)
                else:
                    x_grad = sync_mp_allreduce(task, x_grad)
            return x_grad, None, None
        else:
            if weight.grad is not None:
                assert bias.grad is not None
                weight.grad, bias.grad = _C_ops.fused_linear_param_grad_add(
                    x,
                    y_grad,
                    weight.grad,
                    bias.grad,
                    False if weight.grad.dtype != paddle.float32 else True,
                    True,
                )
                if task is not None:
                    if _sp_async_reduce_scatter:
                        x_grad = sync_sp_reducescatter(task, x_grad)
                    else:
                        x_grad = sync_mp_allreduce(task, x_grad)
                return x_grad, None, None
            else:
                weight_grad, bias_grad = _C_ops.fused_linear_param_grad_add(
                    x, y_grad, None, None, False, True
                )
                if task is not None:
                    if _sp_async_reduce_scatter:
                        x_grad = sync_sp_reducescatter(task, x_grad)
                    else:
                        x_grad = sync_mp_allreduce(task, x_grad)
                return x_grad, weight_grad, bias_grad


class OverlapLinear(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None):
        y = origin_linear(x, weight, bias)
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, y_grad):
        x, weight, bias = ctx.saved_tensor()
        x_grad = paddle.matmul(y_grad, weight, transpose_y=True)

        if _sp_async_reduce_scatter:
            task = sp_async_reducesctter(x_grad)
        else:
            task = mp_async_allreduce(x_grad)

        if _sp_async_reduce_scatter:
            y_grad = dist_reshape(y_grad)
        else:
            y_grad = y_grad.reshape([-1, y_grad.shape[-1]])
        weight_grad = paddle.matmul(
            (
                dist_reshape(x)
                if _sp_async_reduce_scatter
                else x.reshape([-1, x.shape[-1]])
            ),
            y_grad,
            transpose_x=True,
        )
        if bias is None:
            if task is not None:
                if _sp_async_reduce_scatter:
                    x_grad = sync_sp_reducescatter(task, x_grad)
                else:
                    x_grad = sync_mp_allreduce(task, x_grad)
            return x_grad, weight_grad
        else:
            bias_grad = paddle.sum(y_grad, axis=0)
            if task is not None:
                if _sp_async_reduce_scatter:
                    x_grad = sync_sp_reducescatter(task, x_grad)
                else:
                    x_grad = sync_mp_allreduce(task, x_grad)
            return x_grad, weight_grad, bias_grad


def mock_layers(
    enable_fused_linear_grad_add=True,
    enable_mp_async_allreduce=False,
    enable_sp_async_reduce_scatter=False,
):
    global _mp_async_allreduce
    global _sp_async_reduce_scatter
    _mp_async_allreduce = enable_mp_async_allreduce
    _sp_async_reduce_scatter = enable_sp_async_reduce_scatter

    if enable_fused_linear_grad_add:
        paddle.nn.functional.linear = FusedLinearWithGradAdd.apply
        if is_fused_matmul_bias_supported():
            paddle.incubate.nn.functional.fused_linear = (
                FusedLinearWithGradAdd.apply
            )
    else:
        paddle.nn.functional.linear = OverlapLinear.apply
        if is_fused_matmul_bias_supported():
            paddle.incubate.nn.functional.fused_linear = OverlapLinear.apply

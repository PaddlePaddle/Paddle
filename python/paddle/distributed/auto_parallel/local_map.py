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
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Callable

import paddle
import paddle.distributed as dist
from paddle.utils import flatten, pack_sequence_as

if TYPE_CHECKING:
    from paddle.distributed import ProcessMesh


def local_map(
    func: Callable[..., Any],
    out_placements: list[list[dist.Placement]],
    in_placements: list[list[dist.Placement]] | None = None,
    process_mesh: ProcessMesh | None = None,
    reshard_inputs: bool = False,
) -> Callable[..., Any]:
    """
    The `local_map` API allows users to pass dist_tensors to a function that is written
    to be applied on ``paddle.Tensor`` s. It works by extracting the local components
    of dist_tensors, calling the function, and wrapping the outputs as dist_tensors
    according to the ``out_placements``.

    Args:
        func (Callable): The function to be applied on each local shard of dist_tensors.

        out_placements (list[list[dist.Placement]]):
            The desired placements for each output tensor. Must be a list where each element
            is a list of Placement objects specifying the distribution strategy for that
            output tensor. The length of the outer list must match the number of outputs
            from ``func``. For non-tensor outputs, the corresponding placement must be None.
            When there are no dist_tensor inputs, process_mesh must be specified to use
            non-None placements.

        in_placements (Optional[list[list[dist.Placement]]], optional):
            The required placements for each input tensor. If specified, must be a list
            where each element is a list of Placement objects defining the distribution
            strategy for that input tensor. The length of the outer list must match the
            number of input tensors.
            Default: None

        process_mesh (ProcessMesh, optional):
            The process mesh that all dist_tensors are placed on. If not specified,
            this will be inferred from the input dist_tensors' process mesh.
            local_map requires all dist_tensors to be placed on the same process mesh.
            Must be specified when there are no dist_tensor inputs but out_placements
            contains non-None values.
            Default: None

        reshard_inputs (bool, optional):
            the bool value indicating whether to reshard the input :dist_tensors when
            their placements are different from the required input placements. If this
            value is ``False`` and some :dist_tensor input has a different placement,
            an exception will be raised. Default: False.

    Returns:
        Callable: A function that applies func to local shards of input dist_tensors and returns dist_tensors or original values.

    Example:
        .. code-block:: python

            >>> from __future__ import annotations
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> from paddle import Tensor
            >>> from paddle.distributed import ProcessMesh

            >>> def custom_function(x):
            ...     mask = paddle.zeros_like(x)
            ...     if dist.get_rank() == 0:
            ...         mask[1:3] = 1
            ...     else:
            ...         mask[4:7] = 1
            ...     x = x * mask
            ...     mask_sum = paddle.sum(x)
            ...     mask_sum = mask_sum / mask.sum()
            ...     return mask_sum

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> dist.init_parallel_env()
            >>> mesh = ProcessMesh([0, 1], dim_names=["x"])
            >>> local_input = paddle.arange(0, 10, dtype="float32")
            >>> local_input = local_input + dist.get_rank()
            >>> input_dist = dist.auto_parallel.api.dtensor_from_local(
            ...     local_input, mesh, [dist.Shard(0)]
            ... )
            >>> wrapped_func = dist.local_map(
            ...     custom_function,
            ...     out_placements=[[dist.Partial(dist.ReduceType.kRedSum)]],
            ...     in_placements=[[dist.Shard(0)]],
            ...     process_mesh=mesh
            ... )
            >>> output_dist = wrapped_func(input_dist)

            >>> local_value = output_dist._local_value()
            >>> gathered_values: list[Tensor] = []
            >>> dist.all_gather(gathered_values, local_value)

            >>> print(f"[Rank 0] local_value={gathered_values[0].item()}")
            [Rank 0] local_value=1.5
            >>> print(f"[Rank 1] local_value={gathered_values[1].item()}")
            [Rank 1] local_value=6.0
            >>> print(f"global_value (distributed)={output_dist.item()}")
            global_value (distributed)=7.5

            >>> # This case needs to be executed in a multi-card environment
            >>> # export CUDA_VISIBLE_DEVICES=0,1
            >>> # python -m paddle.distributed.launch {test_case}.py
    """

    def wrapped(process_mesh: ProcessMesh | None, *args, **kwargs):
        # Process input arguments
        flat_dist_args = flatten(args)
        if in_placements is not None:
            assert len(in_placements) == len(flat_dist_args), (
                f"in_placements length {len(in_placements)} does not match "
                f"number of input args {len(flat_dist_args)}!"
            )

        flat_local_args = []
        seen_dist_tensor = False

        for idx, arg in enumerate(flat_dist_args):
            if dist.auto_parallel.api.is_dist_tensor(arg):
                dist_tensor = arg
                if process_mesh is None:
                    if paddle.in_dynamic_mode():
                        process_mesh = dist_tensor.process_mesh
                    else:
                        process_mesh = dist_tensor.dist_attr().process_mesh

                seen_dist_tensor = True

                if in_placements is not None:
                    in_placement = in_placements[idx]
                    if in_placement is None:
                        if paddle.in_dynamic_mode():
                            in_placement = dist_tensor.placements
                        else:
                            in_placement = dist_tensor.dist_attr().placements
                    else:
                        if paddle.in_dynamic_mode():
                            if in_placement != dist_tensor.placements:
                                if reshard_inputs:
                                    dist_tensor = dist.reshard(
                                        dist_tensor, process_mesh, in_placement
                                    )
                                else:
                                    raise ValueError(
                                        f"in_placement {in_placement} does not match dist_tensor.placements {dist_tensor.placements}"
                                    )

                        else:
                            if (
                                in_placement
                                != dist_tensor.dist_attr().placements
                            ):
                                if reshard_inputs:
                                    dist_tensor = dist.reshard(
                                        dist_tensor, process_mesh, in_placement
                                    )
                                else:
                                    raise ValueError(
                                        f"in_placement {in_placement} does not match dist_tensor.dist_attr().placements {dist_tensor.dist_attr().placements}"
                                        "If reshard_inputs is wanted, set "
                                        "reshard_inputs=True to local_map."
                                    )
                local_tensor = dist.auto_parallel.api.dtensor_to_local(
                    dist_tensor, process_mesh, in_placement
                )
                flat_local_args.append(local_tensor)
            else:
                flat_local_args.append(arg)

        local_args = pack_sequence_as(args, flat_local_args)
        out = func(*local_args, **kwargs)
        original_out = out
        if seen_dist_tensor:
            flat_out = flatten(out)
            assert len(flat_out) == len(out_placements), (
                "local_map requires one PlacementType for each output value, "
                f"got {len(out_placements)} placements but expected "
                f"{len(flat_out)}!"
            )

            flat_dist_and_arg_out = []
            for out, out_placement in zip(flat_out, out_placements):
                if paddle.in_dynamic_mode():
                    if isinstance(out, paddle.Tensor):
                        assert not dist.auto_parallel.api.is_dist_tensor(
                            out
                        ), f"Expected dense tensor output but got {type(out)}: {out}"

                        flat_dist_and_arg_out.append(
                            dist.auto_parallel.api.dtensor_from_local(
                                out, process_mesh, out_placement
                            )
                        )
                    else:
                        assert out_placement is None, (
                            f"Expected None placements for non-tensor output {out} "
                            f"but got {out_placement}!"
                        )
                        flat_dist_and_arg_out.append(out)
                else:
                    if isinstance(out, paddle.base.libpaddle.pir.Value):
                        assert not dist.auto_parallel.api.is_dist_tensor(
                            out
                        ), f"Expected dense tensor output but got {type(out)}: {out}"

                        flat_dist_and_arg_out.append(
                            dist.auto_parallel.api.dtensor_from_local(
                                out, process_mesh, out_placement
                            )
                        )
                    else:
                        assert out_placement is None, (
                            f"Expected None placements for non-tensor output {out} "
                            f"but got {out_placement}!"
                        )
                        flat_dist_and_arg_out.append(out)
            return pack_sequence_as(original_out, flat_dist_and_arg_out)
        else:
            flat_out = flatten(out)
            flat_dist_and_arg_out = []
            for out, out_placement in zip(flat_out, out_placements):
                if out_placement is not None:
                    assert (
                        process_mesh is not None
                    ), "process_mesh must be specified when out_placements is not None"
                    flat_dist_and_arg_out.append(
                        dist.auto_parallel.api.dtensor_from_local(
                            out, process_mesh, out_placement
                        )
                    )
                else:
                    flat_dist_and_arg_out.append(out)
            return pack_sequence_as(original_out, flat_dist_and_arg_out)

    return functools.partial(wrapped, process_mesh)

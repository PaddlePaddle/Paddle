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

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

import paddle

from .utils import _map_debug_info

logger = logging.getLogger(__name__)


def stage_backward_input(
    stage_outputs_or_loss: list[paddle.Tensor],
    output_grads: list[paddle.Tensor] | None,
    input_values: list[paddle.Tensor],
    weights: Iterator[paddle.Tensor],
) -> tuple[tuple[paddle.Tensor | None, ...], list[dict[str, Any]]]:
    raise NotImplementedError("stage_backward_input is not implemented yet")


def stage_backward_weight(
    weights: Iterator[paddle.Tensor],
    param_groups: list[dict[str, Any]],
    retain_graph=False,
) -> tuple[paddle.Tensor | None, ...]:
    raise NotImplementedError("stage_backward_weight is not implemented yet")


def stage_backward(
    stage_output,
    output_grads,
    input_values,
) -> tuple[paddle.Tensor | None, ...]:
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values

    """

    try:
        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors: list[paddle.Tensor] = []
        output_grad_tensors: list[paddle.Tensor | None] = []

        def extract_tensors_with_grads(
            output_val,
            grad_val,
            extract_tensors_with_grads,
        ):
            if isinstance(output_val, paddle.Tensor):
                if output_val.stop_gradient and output_val.grad_fn is None:
                    return
                assert isinstance(
                    grad_val, (paddle.Tensor, type(None))
                ), f"Expected Tensor or None gradient but got {type(grad_val)}"
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                assert isinstance(
                    grad_val, (tuple, list)
                ), f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                assert len(output_val) == len(grad_val)
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(
                        ov,
                        gv,
                        extract_tensors_with_grads,
                    )
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                for k in output_val.keys():
                    extract_tensors_with_grads(
                        output_val[k], grad_val[k], extract_tensors_with_grads
                    )
            else:
                # Output is a non-tensor type; just ignore it
                pass

        # Note: ref cycle
        # break a ref cycle that would keep tensors alive until GC runs
        # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
        #    and used in extract_tensors_with_grads
        # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
        #    and to itself (extract_tensors_with_grads) since it makes a recursive call
        # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
        # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
        extract_tensors_with_grads(
            stage_output, output_grads, extract_tensors_with_grads
        )
        # Deactivate auto mixed precision context in the backward phase
        with paddle.amp.auto_cast(enable=False):
            paddle.autograd.backward(
                stage_output_tensors, grad_tensors=output_grad_tensors  # type: ignore[arg-type]
            )

        # Extract gradients wrt the input values
        grad_inputs: list[paddle.Tensor | None] = []
        for val in input_values:
            if isinstance(val, paddle.Tensor):
                grad_inputs.append(val.grad)
            else:
                grad_inputs.append(None)

    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {_map_debug_info(stage_output)}
        Output gradient: {_map_debug_info(output_grads)}
        Input: {_map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e

    return tuple(grad_inputs)

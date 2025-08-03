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
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Union

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.base.framework import EagerParamBase
from paddle.distributed.auto_parallel.api import (
    dtensor_from_local,
    dtensor_to_local,
)
from paddle.distributed.communication.group import Group

from ._backward import stage_backward
from .utils import (
    TensorMeta,
    _detach_and_requires_grad,
    _flatten_args,
    _get_stage_mesh,
    _map_debug_info,
    _map_structure_only,
    _validate_tensors_metadata,
    _zero_initialize_with_meta,
    map_structure,
)

logger = logging.getLogger(__name__)


def _restore_placements_info(args, infos, curr_mesh):
    if isinstance(args, paddle.Tensor) and infos.placements is not None:
        # set the placements attribute of the Tensor
        if args.is_dist():
            return _detach_and_requires_grad(args)
        else:
            args = dtensor_from_local(args, curr_mesh, infos.placements)
        return _detach_and_requires_grad(args)
    elif isinstance(args, (list, tuple)):
        # if args is list or tuple, handle each element recursively
        return type(args)(
            _restore_placements_info(a, i, curr_mesh)
            for a, i in zip(args, infos)
        )
    elif isinstance(args, dict):
        # if args is dict, recursively handle each key-value pair
        for key in args:
            _restore_placements_info(args[key], infos[key], curr_mesh)
    else:
        # return directly
        return args


def _normalize_model_output_as_tuple(output: Any) -> tuple[Any]:
    """[Note: pipeline model output type]

    The output of the model passed to pipelining can be any type, controlled by the user.

    However, there are 2 API surfaces that complicate this.
    (1) the outputs of intermediate stages are passed via Send/Recv ops to subsequent stages. The implicit assumption
    is that each element of the outputs is a tensor.  Otherwise, Send/Recv would not be supported.  The exception
    is the last layer of the model, which can output anything any which won't be communicated via Send/Recv.
    (2) the outputs of the last layer of the model are returned to the user, or, passed to the loss function.
    The loss function can be written in any way, such that its inputs match the outputs of the model.

    It would be convenient if we could strictly type the output signature of the pipeline stage wrapping the model,
    but we do not want to impose an unnecessary constraint on user provided models.
    """
    if type(output) is list:
        output = tuple(output)

    # Unify output form to tuple for easy correspondence with
    # `act_send_info`
    output_tuple = output if type(output) is tuple else (output,)
    return output_tuple


class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """

    def __init__(self, tensormeta: TensorMeta):
        self.meta = tensormeta


class _RecvInfo:
    """
    Represents a stage input which is DenseTensor.
    """

    def __init__(
        self,
        input_name: str,
        source: int,
        buffer: paddle.Tensor,
    ):
        # Name of this input
        self.input_name = input_name
        # Stage index of the source of this input
        self.source = source
        # Buffer to receive the input into.
        self.buffer = buffer

    def __repr__(self):
        return f"_RecvInfo(input_name={self.input_name}, source={self.source}, buffer={self.buffer.size})"


# An input can be either a received activation or a model input
InputInfo = Union[_RecvInfo, _RootArgPlaceholder]


def _make_tensor_from_meta(
    example: paddle.Tensor | TensorMeta,
) -> paddle.Tensor:
    """
    Create a real dense tensor from a tensor.
    """
    return paddle.empty(
        example._local_shape if example._local_shape else example.shape,
        dtype=example.dtype,
    )


class _PipelineStageBase(ABC):
    """
    Base class for pipeline stages.
    Defines or implements methods used by manual frontend.
    """

    def __init__(
        self,
        layer: paddle.nn.Layer,
        stage_index: int,
        num_stages: int,
        group: Group | None = None,
    ):
        """
        Args:
            layer (paddle.nn.Layer): The Layer to be executed in this stage.
            stage_index (int): The index of this stage.
            num_stages (int): The total number of stages in this pipeline.
            group (Group|None): The process group to use for communication.
                If `None`, the default process group will be used.
                Default: `None`.
        """
        super().__init__()
        if stage_index >= num_stages:
            raise ValueError(
                f"Stage index {stage_index} is out of range of {num_stages}"
            )

        self.sublayer = layer
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group = group

        # backward state
        self.backward_state: dict[int, tuple[Any, ...]] = {}

        # store dw_runner per microbatch_id
        self.dw_runner: dict[int, Callable[..., None]] = {}

        # `group_rank` is rank in process group `group`.
        self.group_rank = dist.get_rank(self.group)
        self.group_size = dist.get_world_size(self.group)
        if self.group_size > self.num_stages:
            raise RuntimeError(
                f"Pipeline group size {self.group_size} cannot be larger than number of stages {self.num_stages}"
            )

        # Run time states
        self._outputs_meta: tuple[paddle.Tensor, ...] | None = None
        # map microbatch ID to list of forward tensor args
        self.fwd_cache: dict[int, tuple[Any, list[paddle.Tensor]]] = {}
        # map microbatch ID to list of backward grad tensor args
        self.bwd_cache: dict[int, tuple[paddle.Tensor | None, ...]] = {}
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks: list[Any] = []

        # Initialize has_backward to false; this will be set to true if loss
        # function is passed to pipeline schedule
        self.has_backward = False
        # Log prefix
        self.log_prefix = f"[Stage {self.stage_index}]"

        # Forward infra
        self.args_recv_info: dict[int, tuple[InputInfo, ...]] = {}
        self.act_send_info: dict[int, list] = {}
        self._need_grad_indices: dict[int, list] = (
            {}
        )  # record the index of output that needs to receive grad from the next stage.
        # Backward infra will created lazily
        self.grad_recv_info: dict = {}
        self.grad_send_info: list | None = None

        # To be populated later by the Schedule
        self.chunks: int | None = None
        # For V-style pipeline, the calculation of self.stage_index_to_group_rank is not correct here.
        self.stage_index_to_group_rank: dict[int, int] = {
            i: i % self.group_size for i in range(self.num_stages)
        }

    @property
    def has_backward(self) -> bool:
        """
        Returns true if this stage has a backward pass.
        """
        return self._has_backward

    @has_backward.setter
    def has_backward(self, has_backward: bool):
        self._has_backward = has_backward

    @property
    def is_first(self):
        """
        Returns true if this stage is the first stage in the pipeline.
        """
        return self.stage_index == 0

    @property
    def is_last(self):
        """
        Returns true if this stage is the last stage in the pipeline.
        """
        return self.stage_index == self.num_stages - 1

    def _check_chunk_id(self, chunk_id: int):
        if self.chunks is None:
            raise RuntimeError(
                "Attempted to access chunk_id before chunks have been configured."
            )
        if chunk_id >= self.chunks:
            raise RuntimeError(
                f"Chunk id {chunk_id} is out of range [0, {self.chunks})"
            )

    def _configure_outputs_meta(self, outputs_meta: tuple[paddle.Tensor, ...]):
        """
        Track the output shapes/dtype of this stage since they determine the send operation(s) which must match
        recv operations of the next stage.  The next stage _will_ be freezing its recv buffers based on its initial
        configuration, so it's important to also freeze/validate the output side to avoid any send/recv mismatches
        which could show up as hangs, silent corruption, or other errors.
        """
        assert (
            self._outputs_meta is None
        ), "Attempting to reconfigure output_meta, which is not supported"
        self._outputs_meta = tuple(outputs_meta)  # type: ignore[assignment]

    def get_outputs_meta(self) -> tuple[paddle.Tensor, ...]:
        """Get the output metadata (meta tensors) representing the outputs of this stage"""
        assert (
            self._outputs_meta is not None
        ), "Attempted to get_outputs_meta() without configuring output meta"
        return self._outputs_meta

    def _create_grad_send_info(
        self,
        args_recv_info: tuple,
    ) -> list[int | None]:
        """
        Create a list of stage indices to send gradients to.
        """
        grad_send_info: list[int | None] = []

        def map_recv_to_send(a):
            # Note: we send gradients back to previous stage as long as in
            # forward it is a received input, regardless of whether it requires
            # grad. It is up to the previous stage to disgard this gradient.
            if isinstance(a, _RecvInfo):
                grad_send_info.append(a.source)
                return a.source
            else:
                grad_send_info.append(None)
                return None

        map_structure(map_recv_to_send, args_recv_info)

        logger.debug("%s Grad send info: %s", self.log_prefix, grad_send_info)
        return grad_send_info

    @abstractmethod
    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    def _prepare_backward_infra(
        self, num_microbatches: int, loss=None
    ) -> tuple[Any, ...]:
        raise NotImplementedError

    @abstractmethod
    def _create_grad_recv_info(
        self,
        act_send_info: dict,
    ) -> tuple[_RecvInfo, ...]:
        raise NotImplementedError

    def _get_recv_ops(
        self,
        recv_infos: tuple[InputInfo, ...],
    ) -> list[dist.P2POp]:
        """
        Helper function shared by `get_fwd_recv_ops` and `get_bwd_recv_ops`.
        Returns a list of ops that correspond to the recv infos.
        """
        ops: list[dist.P2POp] = []
        for info in recv_infos:
            if not isinstance(info, _RecvInfo):
                continue

            peer_rank = self.stage_index_to_group_rank[info.source]
            peer_global_rank = (
                peer_rank
                if self.group is None
                else self.group.get_global_rank(peer_rank)
            )
            ops.append(
                dist.P2POp(
                    dist.irecv, info.buffer, peer_global_rank, self.group
                )
            )

        return ops

    """[Note: V-schedule special case]

    V-Schedules have a special case where 2 stages with adjacent stage_id are on the same rank.

    ex: 2 ranks, 4 stages forms a simple V:
    rank0:  stage 0                   stage 3
    rank1:          stage 1  stage 2

    stage 0,1 and 2,3 communicate activations using send/recv as usual, but stage 1,2 do not need to
    use communication ops.  Instead, they should pass tensor data directly via function call.

    set_local_fwd_input and (get_local_bwd_output + set_local_bwd_input) facilitate this optimization, and
    should be called at the appropriate time during the pipeline schedule (after forward or backward execution).
    """

    def set_local_fwd_input(
        self, prev_stage_outputs: Any, mb_index: int
    ) -> None:
        """
        Moves 'prev_stage_outputs' from another stage on the same rank into place as inputs for this stage. Avoids
        copying tensor data or using send/recv op.  Detaches original tensor and sets stop_gradient so the
        tensor can serve as a leaf for autograd and gradients can be collected from it during backward.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[mb_index]

        # See [Note: pipeline model output type]
        prev_stage_outputs = _normalize_model_output_as_tuple(
            prev_stage_outputs
        )

        for info, tensor in zip(recv_infos, prev_stage_outputs):
            assert isinstance(
                tensor, paddle.Tensor
            ), f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            assert isinstance(
                info, _RecvInfo
            ), "set_local_Fwd_input should only be called on non-first stage, which should always have RecvInfo"

            info.buffer = _detach_and_requires_grad(tensor)

    def get_local_bwd_output(self, mb_index):
        """
        Returns the input grad tensors for this stage, which correspond to the stage inputs during forward.
        """
        assert (
            self.has_backward
        ), "can't steal_bwd_input if this stage doesn't have backward"
        assert not self.is_first, "can't get bwd output if this stage is first"

        self._check_chunk_id(mb_index)
        return self.bwd_cache.pop(mb_index)

    def set_local_bwd_input(
        self,
        next_stage_bwd_outputs: tuple[paddle.Tensor | None, ...],
        mb_index: int,
    ) -> None:
        """
        Moves 'grad input' tensors from the next stage to 'grad_output' on this stage, avoiding a copy or send/recv.
        Does not detach or set 'stop_gradient'.
        """
        assert isinstance(
            next_stage_bwd_outputs, tuple
        ), f"Expected tuple, got {type(next_stage_bwd_outputs)}"

        assert (
            self.has_backward
        ), "can't set bwd input if this stage doesn't have backward"
        assert not self.is_last, "can't set bwd input if this stage is last"
        recv_infos = self.grad_recv_info[mb_index]
        for info, tensor in zip(recv_infos, next_stage_bwd_outputs):
            assert isinstance(
                tensor, paddle.Tensor
            ), f"expected tensor values as outputs from prev stage, got {type(tensor)}"
            assert isinstance(
                info, _RecvInfo
            ), f"Expected a recv info, got {type(info)}"
            info.buffer = tensor

    def get_fwd_recv_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the input arguments
        for this stage.
        """
        recv_infos: tuple[InputInfo, ...] = self.args_recv_info[fwd_chunk_id]

        return self._get_recv_ops(recv_infos)

    def get_bwd_recv_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Returns a list of ops that are needed to receive the gradients
        for this stage.
        """
        if not self.has_backward or self.is_last:
            return []

        recv_infos = self.grad_recv_info[bwd_chunk_id]
        return self._get_recv_ops(recv_infos)

    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the activation send ops for current stage's forward.
        """
        output = self.output_chunks[fwd_chunk_id]
        # Unify output form to tuple for easy correspondence with
        # `act_send_info`
        output_tuple = output if type(output) is tuple else (output,)

        ops: list[dist.P2POp] = []

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            for dst in dst_stages:
                if dst is None:
                    continue
                logger.debug(
                    "%s Sending tensor to Stage %s: %s",
                    self.log_prefix,
                    dst,
                    out.size,
                )

                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else self.group.get_global_rank(peer_rank)
                )
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        (
                            out
                            if not out.is_dist()
                            else dtensor_to_local(
                                out,
                                out.process_mesh,
                                out.placements,
                            )
                        ),
                        peer_global_rank,
                        self.group,
                    )
                )

        return ops

    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        """
        Get the gradient send ops for current stage's backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        if not self.has_backward or self.is_first:
            return []

        # Create bwd send infra lazily
        if self.grad_send_info is None:
            # Send info for input grads during backward:
            # list of destinations corresponding to input grads
            # Can be None if an input has no grad
            # `grad_send_info` is a mirror of `args_recv_info`
            self.grad_send_info = self._create_grad_send_info(
                self.args_recv_info[0]
            )

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)
        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, paddle.Tensor) and grad_recv_stage is not None:
                logger.debug(
                    "%s Sending gradient to Stage %s: %s",
                    self.log_prefix,
                    grad_recv_stage,
                    grad.size,
                )
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else self.group.get_global_rank(peer_rank)
                )
                ops.append(
                    dist.P2POp(
                        dist.isend,
                        (
                            grad
                            if not grad.is_dist()
                            else dtensor_to_local(
                                grad, grad.process_mesh, grad.placements
                            )
                        ),
                        peer_global_rank,
                        self.group,
                    )
                )
            else:
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradients {grad} "
                        f"and is expecting to send gradients to stage {grad_recv_stage}"
                    )
        return ops

    def clear_runtime_states(self) -> None:
        """
        Clear runtime states of the stage.
        """
        # map microbatch ID to list of forward tensor args
        self.fwd_cache.clear()
        # Caching chunk outputs for final output merge or reduction
        self.output_chunks.clear()

        # Clear grad of input buffers in between schedule steps. This is because
        # `paddle.autograd.backward()` will accumulate gradients into leaf
        # tensors by default. For gradients to pass back to previous stages, we
        # don't want such accumulation.
        for (
            recv_tuple
        ) in self.args_recv_info.values():  # iterate over all chunks
            for a in recv_tuple:  # iterate over all input args
                if isinstance(a, _RecvInfo):
                    a.buffer.clear_grad()

    def _map_tensor_from_recv_info(
        self,
        recv_infos: tuple[InputInfo, ...],
    ):
        """
        Map tensors from recv infos to a list.
        """

        def get_recv_tensor(info):
            if isinstance(info, _RecvInfo):
                return info.buffer
            else:
                raise AssertionError(f"Expected _RecvInfo but got {type(info)}")

        tensors = map_structure(
            get_recv_tensor,
            recv_infos,  # type: ignore[arg-type]
        )

        return tensors

    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        """
        Retrieve the activations received for the current stage during forward.
        """
        recv_infos = self.args_recv_info[fwd_chunk_id]
        activations = self._map_tensor_from_recv_info(recv_infos)
        return activations

    def _retrieve_recv_grads(
        self,
        bwd_chunk_id: int,
    ):
        """
        Retrieve the gradients received for the current stage during backward.
        """
        recv_infos = self.grad_recv_info[bwd_chunk_id]
        grads = self._map_tensor_from_recv_info(recv_infos)
        return grads

    def forward_maybe_with_nosync(self, *args, **kwargs):
        curr_mesh = _get_stage_mesh(self.stage_index, self.group_size)

        args = _restore_placements_info(args, self.inputs_meta, curr_mesh)

        out_val = self.sublayer(*args, **kwargs)

        return out_val, args

    def backward_maybe_with_nosync(
        self, backward_type, bwd_kwargs: dict, last_backward=False
    ) -> tuple[tuple[paddle.Tensor | None, ...], list[dict[str, Any] | None]]:

        def perform_backward(
            backward_type,
        ) -> Callable[
            [],
            tuple[
                tuple[paddle.Tensor | None, ...],
                list[dict[str, Any] | None],
            ],
        ]:
            if backward_type == "full":
                return lambda: (
                    stage_backward(
                        bwd_kwargs["stage_output"],
                        bwd_kwargs["output_grads"],
                        bwd_kwargs["input_values"],
                    ),
                    None,
                )
            elif backward_type == "input":
                raise NotImplementedError(
                    "Input based backward is not implemented yet."
                )
            elif backward_type == "weight":
                raise NotImplementedError(
                    "Weight based backward is not implemented yet."
                )
            else:
                raise RuntimeError(f"Unknown backward type: {backward_type}")

        curr_mesh = _get_stage_mesh(self.stage_index, self.group_size)
        bwd_kwargs["output_grads"] = _restore_placements_info(
            bwd_kwargs["output_grads"], self.grads_meta, curr_mesh
        )
        result = perform_backward(backward_type)()
        grads, param_groups = result
        return grads, param_groups

    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Perform forward pass on the stage with one microbatch.
        `args` and `kwargs` are the inputs from *external* to this stage.
        - `args` applies to the first stage only, other stages receives args
          through activation transmission.
        - `kwargs` can be passed to all stages via respective `step` calls.
        """

        if self.is_first:
            # First stage doesn't need to receive anything
            composite_args = args
        else:
            # Receive activations for this chunk
            # Activations only come in args form
            composite_args = self._retrieve_recv_activations(fwd_chunk_id)

        composite_kwargs = kwargs or {}

        self._validate_fwd_input(args, kwargs)

        # Compute forward
        try:
            output, input_args = self.forward_maybe_with_nosync(
                *composite_args, **composite_kwargs
            )

        except Exception as e:
            exc_msg = f"""
            {self.log_prefix} failed to run forward:
            args: {_map_debug_info(composite_args)}
            kwargs: {_map_debug_info(composite_kwargs)}
            """
            raise RuntimeError(exc_msg) from e

        # See [Note: pipeline model output type]
        output_tuple = _normalize_model_output_as_tuple(output)

        # Prepare for final output merge or reduction
        self.output_chunks.append(output)

        # Save activations and inputs for backward
        flat_args = _flatten_args(input_args)
        flat_kwargs = _flatten_args(composite_kwargs)
        flatten_input_tensors = flat_args + flat_kwargs
        requires_grad_output_tuple = tuple(
            out
            for out in output_tuple
            if isinstance(out, paddle.Tensor) and not out.stop_gradient
        )
        flatten_requires_grad_input_tensors = [
            inp
            for inp in flatten_input_tensors
            if isinstance(inp, paddle.Tensor) and not inp.stop_gradient
        ]
        self.fwd_cache[fwd_chunk_id] = (
            requires_grad_output_tuple,  # stage_output
            flatten_requires_grad_input_tensors,  # input_values
        )

        logger.debug(
            "%s Forwarded chunk %s, outputs: %s",
            self.log_prefix,
            fwd_chunk_id,
            _map_debug_info(output),
        )
        self._validate_fwd_outputs(output_tuple)

        # We return the original user-provided output, not normalized to tuple.
        # See [Note: pipeline model output type]
        return output

    def backward_one_chunk(
        self,
        bwd_chunk_id: int,
        loss=None,
        full_backward: bool = True,
        last_backward=False,
    ):
        """
        Perform backward pass on the module.
        This should only be called once per microbatch.

        If full_backward is True (the default), the full backward pass including weight and input gradients will be run,
        and it is an error to call `backward_weight_one_chunk` for this bwd_chunk_id.

        If full_backward is False, it is optional that `dw_runner` was provided to the PipelineStage at __init__ time,
        and a subsequent call to `backward_weight_one_chunk` is required to invoke dw_runner and complete the backward.

        last_backward is controlled by the schedule and signals synchronization of gradients across DP groups
        after the last backward.
        """
        self._check_chunk_id(bwd_chunk_id)

        (
            stage_output,
            input_values,
        ) = self.fwd_cache.pop(bwd_chunk_id)

        # Compute backward
        if self.is_last:
            # Last stage computes gradients from loss and has no gradients from
            # next stage
            bwd_kwargs = {
                "stage_output": loss,
                "output_grads": None,
                "input_values": input_values,
            }
        else:
            # Otherwise, receive gradients from next stage
            grads_output = self._retrieve_recv_grads(bwd_chunk_id)
            # If an input to the pipeline requires gradient,
            # `paddle.autograd.backward` will accumulate the gradient into the
            # `.grad` field of such input
            bwd_kwargs = {
                "stage_output": stage_output,
                "output_grads": grads_output,
                "input_values": input_values,
            }

        grads_input: tuple[paddle.Tensor | None, ...] = ()

        if full_backward:
            grads_input, _ = self.backward_maybe_with_nosync(
                "full", bwd_kwargs, last_backward=last_backward
            )
        else:
            raise NotImplementedError(
                "Input based backward is not implemented yet."
            )

        self.bwd_cache[bwd_chunk_id] = grads_input
        if self.is_last and not self.is_first:
            # Autograd dependencies:
            #    rest_of_autograd_graph -> stage_output -> loss
            # stage_output is no longer used in the last stage for backward and only needed
            # to return to the user in merge_output_chunks, therefore
            # this should be detached to release autograd graph context and free memory earlier
            for t in stage_output:
                t.detach_()

        logger.debug("%s Backwarded chunk %s", self.log_prefix, bwd_chunk_id)
        return grads_input

    def backward_weight_one_chunk(self, bwd_chunk_id: int, last_backward=False):
        raise NotImplementedError(
            "Weight based backward is not implemented yet."
        )

    def _validate_fwd_input(self, args, kwargs):
        """Raises a RuntimeError if shapes of input args/kwargs do not match the shapes configured for this stage."""

        if self.is_first:
            expected_args = self.args_recv_info[0]
        else:
            return

        if len(kwargs):
            return

        expected_tensors_meta = [
            e.meta if isinstance(e, _RootArgPlaceholder) else e.buffer
            for e in expected_args
        ]
        _validate_tensors_metadata(
            f"Stage {self.stage_index} forward inputs",
            expected_tensors_meta,
            args,
        )

    def _validate_fwd_outputs(self, outputs: tuple[paddle.Tensor, ...]):
        """Raises a RuntimeError if this stage produces an output of unexpected shape/dtype.
        Most likely, this could be cause either by incorrect user specification of output shapes, or because
        shape inference was done on the original model but then at runtime the model is wrapped with something like
        mixed precision which changes output dtype.
        """
        expected_tensors_meta = self.get_outputs_meta()
        _validate_tensors_metadata(
            f"Stage {self.stage_index} forward outputs",
            expected_tensors_meta,
            outputs,
        )


class PipelineStage(_PipelineStageBase):
    """
    A class representing a pipeline stage in a pipeline parallelism setup.

    PipelineStage assumes sequential partitioning of the model, i.e. the model is split into chunks where outputs from
    one chunk feed into inputs of the next chunk. Additionally, optimization of shared parameters is also supported here.

    PipelineStage performs runtime shape/dtype inference automatically by propagating the outputs from stage0 to
    stage1 and so forth, in linear order.  To bypass shape inference, pass the `input_args` and `output_args` to each
    PipelineStage instance.

    Args:
        layer (nn.Layer): The Pypaddle module wrapped by this stage.
        stage_index (int): The ID of this stage.
        num_stages (int): The total number of stages.
        input_args (TensorMeta|tuple[TensorMeta, ...]|None): The input arguments for the layer.
        output_args (TensorMeta|tuple[TensorMeta, ...]|None): The output arguments for the layer.
        group (Group, None): The process group for distributed training. If None, default group.
        shared_parameters (list[dict[str, list[EagerParamBase]]]|None): A list of dictionaries defining shared parameter
        pairs between pipeline stages. Each dictionary represents a unique parameter pair with:
            - "params" (list[EagerParamBase], required): Exactly 2 parameters to share across stages.
    """

    def __init__(
        self,
        layer: nn.Layer,
        stage_index: int,
        num_stages: int,
        input_args: TensorMeta | tuple[TensorMeta, ...] | None = None,
        output_args: TensorMeta | tuple[TensorMeta, ...] | None = None,
        group: Group | None = None,
        shared_parameters: list[dict[str, list[EagerParamBase]]] | None = None,
    ):
        super().__init__(layer, stage_index, num_stages, group)
        self.inputs: list[paddle.Tensor] | None = None
        self.inputs_meta: tuple[TensorMeta, ...] | None = None
        # output's grad meta-info
        self.grads_meta: tuple[TensorMeta, ...] | None = None

        # Synchronize shared parameters on the current rank.
        self.shared_parameters = shared_parameters
        self._sync_shared_param()

        if input_args is None:
            assert output_args is None, (
                "If specifying output_args, input_args must also be specified. "
                "Otherwise, shape inference will be performed at runtime"
            )
        else:
            self.inputs_meta = (
                (input_args,)
                if isinstance(input_args, TensorMeta)
                else input_args
            )

            assert (
                output_args is not None
            ), "If passing input_args, also pass output_args to override shape inference"
            self._configure_outputs_meta(
                (output_args,)
                if isinstance(output_args, TensorMeta)
                else output_args
            )

        # these are the buffers used in backwards send/recv, they are allocated later
        self.outputs_grad: list[paddle.Tensor] = []

        def stage_global_rank(peer_rank):
            return (
                peer_rank
                if self.group is None
                else group.get_global_rank(peer_rank)
            )

        self.prev_rank = stage_global_rank(
            (self.group_rank - 1) % self.group_size
        )
        self.next_rank = stage_global_rank(
            (self.group_rank + 1) % self.group_size
        )

        dbg_str = (
            f"Finished pipeline stage init, {self.stage_index=}, {self.is_first=}, "
            f"{self.is_last=}, {self.num_stages=}, "
        )
        if self.inputs_meta is not None:
            dbg_str += (
                f"inputs: {[inp.shape for inp in self.inputs_meta]}, "
                f"output: {[output.shape for output in self.get_outputs_meta()]}"
            )
        else:
            dbg_str += " running shape-inference at runtime"

        logger.debug(dbg_str)

    def _sync_shared_param(self):
        if self.shared_parameters is None:
            # 1. Default no shared parameters to process.
            self.shared_parameters = {}
            return

        # 2. Validate parameters.
        # TODO(xuexixi): Currently, shared parameter information relies on user input, so strict validation is required here.
        # A more robust interface implementation is desired in the future.
        self._validate_shared_parameter_pair()

        # 3. Build shared parameter information for the current rank.
        self._init_shared_group()

        # 4. Synchronize the initialized shared parameters.
        # When initializing the stage, perform broadcast synchronization on the shared parameters.
        for idx, a_map in enumerate(self.shared_parameters):
            shared_param = a_map["shared_param"]
            if shared_param is None or not shared_param._is_initialized():
                # Skip processing shared parameters that are not assigned to the current rank.
                continue
            group = a_map.get("group")
            assert group is not None and dist.get_rank() in group.ranks
            logger.debug(
                f"Call `broadcast` for synchronization of Shared parameter pair at index {idx}",
            )
            with paddle.no_grad():
                paddle.distributed.broadcast(
                    shared_param._local_value(),
                    src=group.ranks[0],
                    group=group,
                )

    def _validate_shared_parameter_pair(self):
        # Validate shared_parameters structure.
        assert isinstance(
            self.shared_parameters, list
        ), f"Expected `shared_parameters` to return a list, but got {type(self.shared_parameters).__name__}. "

        # Validate every pair shard parameter.
        for idx, a_shared_map in enumerate(self.shared_parameters):
            # Validate map structure.
            assert isinstance(
                a_shared_map, dict
            ), f"Invalid shared parameter pair: expected dict, but got {type(a_shared_map).__name__}."
            assert len(a_shared_map) <= 3, (
                f"shared_parameters['{idx}'] exceeds size limit (max 3 keys). "
                f"Allowed: ['params', 'group', 'shared_param'], got: {list(a_shared_map.keys())}"
            )
            # Validate required 'params' entry.
            params = a_shared_map.get("params")
            assert (
                params is not None
            ), f"Missing shared parameter 'params' not found in shared_parameters['{idx}']. Available keys: {list(a_shared_map)}."
            assert (isinstance(params, list) or tuple(params, list)) and len(
                params
            ) == 2, f"Shared parameter only support 2 shared parameters in list or tuple, but got {len(params)}."
            # Validate parameter types and placements.
            param_1, param_2 = params
            assert isinstance(param_1, EagerParamBase) and isinstance(
                param_2, EagerParamBase
            ), (
                f"Shared parameter expects parameters are 'EagerParamBase' type, but got "
                f"'{type(param_1).__name__}' and '{type(param_2).__name__}' respectively."
            )
            assert param_1.placements == param_2.placements, (
                f"Shared parameters must have identical placements for optimal performance."
                f"But placements mismatch: {param_1.placements} vs {param_2.placements}"
            )
            # Validate process meshes.
            ranks_1 = param_1.process_mesh.process_ids
            ranks_2 = param_2.process_mesh.process_ids
            assert len(ranks_1) == len(ranks_2)
            assert (
                ranks_1 != ranks_2
            ), f"Shared parameters must be on different stage meshes, but both are on {ranks_1}."

            # In VPP mode, a same shared_parameters is reused across stage builds. To avoid redundant group creation, the 'shared_param'
            # and 'group' attributes may already exist, as they are created during the `_init_shared_group` call.
            # Validate optional 'group' entry.
            if "group" in a_shared_map:
                group = a_shared_map["group"]
                assert group is None or isinstance(
                    group, Group
                ), f"Expected 'shared_parameters[{idx}][\"group\"]' is 'Group' or None, but got '{type(a_shared_map['group']).__name__}'."
            # Validate optional 'sync_param' entry.
            if "sync_param" in a_shared_map:
                sync_param = a_shared_map["sync_param"]
                assert sync_param is None or sync_param in list(
                    param_1, param_2
                ), f"Expected 'shared_parameters[{idx}][\"sync_param\"]' is one of the two params or None."

    def _init_shared_group(self):
        # Retrieve the parameters to be shared and the required communication group information for the current rank, and store them in
        # the 'shared_param' and 'group' attributes of the shared_parameters respectively:
        #   - group (Group, optional): Communication group for sharing the current parameter pair on the current rank (auto-created if missing)
        #   - shared_param (EagerParamBase, optional): Parameter to be shared on the current rank, should be one of 'params'; if None, it means
        #           no sharing is required on this rank. (auto-populated if missing)
        get_group_from_ranks = {}
        for idx, a_map in enumerate(self.shared_parameters):
            params = a_map["params"]
            ranks_1 = params[0].process_mesh.process_ids
            ranks_2 = params[1].process_mesh.process_ids
            cur_rank = dist.get_rank()

            # Build communication groups for every shared parameters pair.
            for rank_1, rank_2 in zip(ranks_1, ranks_2):
                group_ranks = tuple(sorted([rank_1, rank_2]))
                if "group" in a_map:
                    # In VPP mode, since `shared_parameters`` is reused across stage creations,
                    # the 'group' may already exist, avoiding redundant group creation.
                    if cur_rank in group_ranks:
                        assert group_ranks == tuple(
                            a_map["group"].ranks
                        ), f"Shared Parameter group ranks mismatch: expected {group_ranks}, but got {a_map['group'].ranks}. "
                else:
                    if group_ranks not in get_group_from_ranks:
                        get_group_from_ranks[group_ranks] = dist.new_group(
                            ranks=list(group_ranks)
                        )
                    if cur_rank in group_ranks:
                        # Record `group` is communication group associated with the current rank.
                        a_map["group"] = get_group_from_ranks[group_ranks]
                        logger.debug(
                            f"Build communication group {a_map['group'].name} for Shared parameter pair at index {idx} in rank {cur_rank}"
                        )

            # Find the shared parameter on the current rank.
            # Record `shared_param` is None default no shared parameter exists on current rank.
            cur_param = None
            if cur_rank in ranks_1:
                cur_param = params[0]
            elif cur_rank in ranks_2:
                cur_param = params[1]
            # Record shared parameter associated with the current rank.
            a_map["shared_param"] = cur_param

    def _sync_shared_param_grads(self):
        # After the stage scheduling ends, perform allreduce synchronization
        # on the gradients of shared parameters.
        for idx, a_map in enumerate(self.shared_parameters):
            shared_param = a_map["shared_param"]
            if shared_param is None or not shared_param._is_initialized():
                # Skip processing shared parameters that are not assigned to the current rank.
                continue
            group = a_map.get("group")
            assert group is not None and dist.get_rank() in group.ranks
            logger.debug(
                f"Call `all_reduce` for gradient synchronization of Shared parameter pair at index {idx}",
            )
            with paddle.no_grad():
                paddle.distributed.all_reduce(
                    shared_param.grad._local_value(),
                    op=paddle.distributed.ReduceOp.SUM,
                    group=group,
                )

    def _shape_inference(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ):
        if kwargs is None:
            kwargs = {}
        assert args is not None, "Args may be an empty tuple but not None"

        # We skip recv communication if we're the first stage, but also if the previous stage is on the same rank
        # and can pass its output shapes in as args instead of using send/recv.
        if self.is_first:
            logger.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `args`",
                self.stage_index,
            )
            args = _map_structure_only(
                paddle.Tensor,
                lambda x: TensorMeta(x),
                args,
            )
        elif (
            self.stage_index_to_group_rank[self.stage_index - 1]
            == self.group_rank
        ):
            raise NotImplementedError
        else:
            assert (
                len(args) == 0
            ), "Can't supply input args for shape inference on non-first stage"
            objects = [None]
            logger.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.recv_object_list(objects, src=self.prev_rank, group=self.group)
            recv_args = objects[0]
            assert isinstance(recv_args, tuple), type(recv_args)
            args = recv_args

        # cache input shapes for use during recv buffer allocation
        self.inputs_meta = args
        # zero-initialise tensors only for inference outputs
        zero_initialize_with_meta_ = partial(
            _zero_initialize_with_meta,
            mesh=_get_stage_mesh(self.stage_index, self.group_size),
        )
        args = _map_structure_only(
            TensorMeta,
            zero_initialize_with_meta_,
            args,
        )

        # set attributes needed for forward
        with (
            paddle.no_grad() if not self.has_backward else paddle.enable_grad()
        ):
            logger.debug(
                "Shape inference: stage %s running forward", self.stage_index
            )
            if self.has_backward and not self.is_first:

                def requires_grad(x):
                    x.stop_gradient = False
                    return x

                args = _map_structure_only(paddle.Tensor, requires_grad, args)

            outputs = self.sublayer(*args, **kwargs)
            if self.has_backward:
                flatten_input_tensors = _flatten_args(args) + _flatten_args(
                    kwargs
                )
                # cache the forward outputs for backward, so remove tensor that stop_gradient = True
                flatten_input_tensors = [
                    x for x in flatten_input_tensors if not x.stop_gradient
                ]
                grad_required_outputs = _normalize_model_output_as_tuple(
                    outputs
                )
                grad_required_outputs = tuple(
                    out
                    for out in grad_required_outputs
                    if isinstance(out, paddle.Tensor) and not out.stop_gradient
                )
                self.fwd_cache[0] = (
                    grad_required_outputs,  # stage_output
                    flatten_input_tensors,  # input_values
                )

        # if single tensor, convert so it is always a list
        if isinstance(outputs, paddle.Tensor):
            outputs = [outputs]

        # communicate meta outputs not real outputs for two reasons
        # 1 - its faster (esp. since obj coll pickles tensor data!)
        # 2 - avoid activating a cuda context for the src rank when unpickling on the recv end!
        outputs_meta = tuple(
            _map_structure_only(paddle.Tensor, lambda x: TensorMeta(x), outputs)
        )
        self._configure_outputs_meta(outputs_meta)

        # Passing outputs to the next stage:
        # two cases-
        # 1. Usually: use send/recv communication to pass the output
        # 2. Special case: for V-schedules, 2 'adjacent' stages (e.g. stage 3, 4 in an 8-stage 4-rank V)
        #    pass their shape info via return value and function args rather than send/recv.
        if self.is_last:
            # Case (2) above: pass shape info via return value and caller passes it as args to next stage's
            # _shape_inference call
            logger.debug(
                "Shape inference: stage %s skipping send to next stage",
                self.stage_index,
            )
            # keep the origin output of the last stage for backward
            return outputs
            # if not last stage, then check if next stage is on the same rank
        elif (
            self.stage_index_to_group_rank[self.stage_index + 1]
            == self.group_rank
        ):
            raise NotImplementedError
        else:
            # Case (1): send shapes via send operation, and ensure not to return it to the caller
            logger.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.send_object_list(
                [outputs_meta],
                dst=self.next_rank,
                group=self.group,
            )
            outputs_meta = ()

        return outputs_meta

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, ...]:

        assert num_microbatches is not None, "num_microbatches must be provided"

        outputs: tuple[Any, ...] = ()
        if self.inputs_meta is None:
            outputs = self._shape_inference(args, kwargs)

        assert self.inputs_meta is not None

        for chunk_id in range(num_microbatches):
            if not self.is_first:
                # We assume that we always receive from stage - 1
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self.stage_index}_from_{self.stage_index - 1}",
                            self.stage_index - 1,
                            _make_tensor_from_meta(inp),
                        )
                        for inp in self.inputs_meta
                    ]
                )
                # In case there is backward pass, set stop_gradient for receive buffers
                if self.has_backward:
                    for r in recv_infos:
                        r.buffer.stop_gradient = False

                self.args_recv_info[chunk_id] = recv_infos
            else:
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs_meta]
                )

        # Send info during forward for each activation
        # only need the rank that is being sent to
        self.act_send_info: dict[int, list] = {}
        outputs_meta = self.get_outputs_meta()
        for idx in range(len(outputs_meta)):
            # We assume we always send to stage + 1
            if not self.is_last:
                self.act_send_info[idx] = [self.stage_index + 1]
                if not outputs_meta[idx].stop_gradient:
                    self._need_grad_indices[idx] = [self.stage_index + 1]
            else:
                self.act_send_info[idx] = []
                self._need_grad_indices[idx] = []

        return outputs

    def _shape_inference_bwd(
        self,
        loss=None,
    ):
        assert self.fwd_cache is not None
        stage_output, input_values = self.fwd_cache.pop(0)
        if self.is_last:
            assert loss is not None, "loss cannot be none during backward"
            logger.debug(
                "Shape inference: stage %s skipping recv, because shape info passed in via `grads`",
                self.stage_index,
            )
            stage_output = loss
            grads = None
        elif (
            self.stage_index_to_group_rank[self.stage_index + 1]
            == self.group_rank
        ):
            raise NotImplementedError
        else:
            objects = [None]
            logger.debug(
                "Shape inference: stage %s receiving from stage %s",
                self.stage_index,
                self.stage_index + 1,
            )
            dist.recv_object_list(objects, src=self.next_rank, group=self.group)
            recv_grads = objects[0]
            assert isinstance(recv_grads, tuple), type(recv_grads)
            grads = recv_grads

            self.grads_meta = grads

            # zero-initialize tensors only for inference backward meta-info
            zero_initialize_with_meta_ = partial(
                _zero_initialize_with_meta,
                mesh=_get_stage_mesh(self.stage_index, self.group_size),
            )
            grads = _map_structure_only(
                TensorMeta, zero_initialize_with_meta_, grads
            )
        with paddle.amp.auto_cast(enable=False):
            paddle.autograd.backward(stage_output, grads, True)

        # output is the grad meta for input_values(list)
        if not self.is_first:
            output_meta = tuple(
                map_structure(lambda x: TensorMeta(x.grad), input_values)
            )

        if self.is_first:
            logger.debug(
                "Shape inference: stage %s skipping send to previous stage",
                self.stage_index,
            )
        elif (
            self.stage_index_to_group_rank[self.stage_index - 1]
            == self.group_rank
        ):
            raise NotImplementedError
        else:
            logger.debug(
                "Shape inference: stage %s sending to stage %s",
                self.stage_index,
                self.stage_index - 1,
            )
            dist.send_object_list(
                [output_meta], dst=self.prev_rank, group=self.group
            )

    def _prepare_backward_infra(
        self, num_microbatches: int, loss=None
    ) -> tuple[Any, ...]:
        assert self.has_backward is not None

        self.chunks = num_microbatches

        grads: tuple[Any, ...] = ()
        if self.grads_meta is None:
            if self.is_last:
                self._shape_inference_bwd(loss)
            else:
                self._shape_inference_bwd()
        for mb_index in range(num_microbatches):
            # `grad_recv_info` is a mirror of `act_send_info`
            self.grad_recv_info[mb_index] = self._create_grad_recv_info()

        # the last stage does not need recv grads from other rank
        if not self.is_last:
            assert self.grads_meta is not None
        # clear backward_state
        self.clear_runtime_states()
        # Clear grads of the stage.
        for param in self.sublayer.parameters():
            param.clear_grad()
        return grads

    def _create_grad_recv_info(
        self,
    ) -> tuple[_RecvInfo, ...]:
        grad_recv_info: tuple[_RecvInfo, ...] = ()
        if not self.is_last:
            # Receiving gradients from multiple sources is not supported
            # hence we only take the first destination
            grad_recv_info = tuple(
                [
                    _RecvInfo(
                        f"recv_grad_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        _make_tensor_from_meta(self.grads_meta[idx]),
                    )
                    for idx, dst_list in self._need_grad_indices.items()
                ]
            )
        return grad_recv_info

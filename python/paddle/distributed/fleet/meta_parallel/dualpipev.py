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

# The file has been adapted from DeepSeek DualPipe project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DualPipe/blob/main/LICENSE

from __future__ import annotations

import paddle
from paddle import framework
from paddle.distributed.communication.batch_isend_irecv import (
    P2POp,
    batch_isend_irecv,
)

try:
    from paddle.distributed.communication import deep_ep
except ImportError:
    deep_ep = None

from ..utils.log_util import logger
from .pipeline_parallel import (
    FakeMicroDataset,
    HybridParallelOptimizer,
    PipelineParallel,
)
from .pp_utils.batch_comm_helper import BatchCommHelper
from .zero_bubble_utils import WeightGradStore

__all__ = []


def detach_and_requires_grad(x):
    o = x.detach()
    o.stop_gradient = False
    return o


class DualPipeVParallel(PipelineParallel):
    """
    An implementation of the DualPipeV, based on
    https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipe.py.
    """

    def __init__(self, layers, hcg, strategy):
        super().__init__(layers=layers, hcg=hcg, strategy=strategy)
        self.overlapped_forward_backward = hasattr(
            type(self._layers), "overlapped_forward_backward"
        )
        logger.info(
            f"Using DualPipeVParallel with overlapping forward backward={self.overlapped_forward_backward}"
        )

        self.num_ranks = self.num_stages
        self.group_rank = self.pp_group.rank
        self.prev_rank = self.pp_group.ranks[
            (self.group_rank - 1) % self.pp_group.world_size
        ]
        self.next_rank = self.pp_group.ranks[
            (self.group_rank + 1) % self.pp_group.world_size
        ]

        # NOTE(zhangyuqin1998): The first rank has to broadcast the meta information
        # of the P2P communication after the first forward.
        self.need_broadcast_meta = self.is_pipeline_first_stage()
        self.need_recv_meta = not self.is_pipeline_first_stage()
        self._p2p_helper = BatchCommHelper(self._using_cache)

    def is_pipeline_first_stage(self):
        return self.group_rank == 0

    def is_pipeline_last_stage(self):
        return self.group_rank == self.num_ranks - 1

    def _reset_states(self):
        self.input_tensors = ([], [])
        self.output_tensors = ([], [])
        self.input_grad_tensors = ([], [])
        self.output_grad_tensors = ([], [])
        self.loss_tensors: list[paddle.Tensor] = []
        self.schedule_chunks = ([], [])
        self.loss_fn_chunks = []

        # The first value in the list corresponds to phase 0, and the second value corresponds to phase 1.
        self.current_f_acc_id = [0, 0]
        self.current_b_acc_id = [0, 0]
        self.current_send_f_acc_id = [0, 0]
        self.current_send_b_acc_id = [0, 0]
        self.current_recv_f_acc_id = [0, 0]
        self.current_recv_b_acc_id = [0, 0]
        self.comm_forward_ops: list[P2POp] = []
        self.comm_backward_ops: list[P2POp] = []
        self.to_free: list[paddle.Tensor] = []

    def _get_forward_inputs(self, micro_datasets, phase, acc_id):
        is_first_stage = self.is_pipeline_first_stage() and phase == 0
        if is_first_stage:
            assert micro_datasets is not None
            self.input_tensors[phase].append(next(micro_datasets[phase])[0])
        if self.forward_only:
            self.input_tensors[phase][acc_id] = None
        return self.input_tensors[phase][acc_id]

    def _get_forward_labels(self, micro_datasets, phase, acc_id):
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            assert micro_datasets is not None
            labels = next(micro_datasets[phase])[1]
            self._check_micro_batch_data_valid(labels)
            return labels
        else:
            return None

    def _loss_compute(self, micro_datasets, phase, acc_id, logits):
        labels = self._get_forward_labels(micro_datasets, phase, acc_id)
        loss_fn_node = None
        if not self.overlapped_forward_backward:
            loss_tensor = self._layers._loss_fn[0](logits, labels)
            with paddle.amp.auto_cast(enable=False):
                if self.accumulate_steps > 1 and not self._delay_scale_loss:
                    loss_tensor = loss_tensor / self.accumulate_steps
        else:
            loss_fn_node = self._layers._loss_fn[0].build_schedule_node()
            loss_fn_node.labels = labels
            if self.accumulate_steps > 1 and not self._delay_scale_loss:
                loss_fn_node.scale_loss_factor = self.accumulate_steps
            loss_tensor = loss_fn_node.forward(logits)
        self._store_forward_loss(phase, loss_tensor, loss_fn_node)

    def _store_forward_tensors(self, phase, outputs, schedule_chunk):
        self.schedule_chunks[phase].append(schedule_chunk)
        if self.is_pipeline_last_stage() and phase == 0:
            self.input_tensors[1].append(
                [detach_and_requires_grad(output) for output in outputs]
            )
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if not is_last_stage:
            self.output_tensors[phase].append(outputs)

    def _forward_compute(self, phase: int, micro_datasets=None) -> None:
        acc_id = self.current_f_acc_id[phase]
        self.current_f_acc_id[phase] += 1

        inputs = self._get_forward_inputs(micro_datasets, phase, acc_id)

        if self.overlapped_forward_backward:
            schedule_chunk = self._layers.get_schedule_chunk(chunk_id=phase)
            outputs = schedule_chunk.forward(inputs)
        else:
            schedule_chunk = None
            outputs = self._layers.forward(inputs, chunk_id=phase)
        outputs = [outputs] if isinstance(outputs, paddle.Tensor) else outputs

        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            self._loss_compute(micro_datasets, phase, acc_id, outputs)
        self._store_forward_tensors(phase, outputs, schedule_chunk)

    def _get_backward_inputs(self, phase, acc_id):
        outputs = self.output_tensors[phase][acc_id]
        self.output_tensors[phase][acc_id] = None
        output_grads = self.output_grad_tensors[phase][acc_id]
        self.output_grad_tensors[phase][acc_id] = None
        non_empty = [
            (t, g) for t, g in zip(outputs, output_grads) if g is not None
        ]
        outputs, output_grads = list(zip(*non_empty))
        return outputs, output_grads

    def _store_backward_tensors(self, phase, acc_id, input_grads=None):
        if input_grads is None:
            inputs = self.input_tensors[phase][acc_id]
            input_grads = [
                t.grad
                for t in inputs
                if (t is not None and not t.stop_gradient)
            ]
        self.input_tensors[phase][acc_id] = None

        if isinstance(input_grads, paddle.Tensor):
            input_grads = (input_grads,)
        if self.is_pipeline_last_stage() and phase == 1:
            self.output_grad_tensors[0].append(input_grads)
        else:
            self.input_grad_tensors[phase].append(input_grads)

    def _store_forward_loss(self, phase, loss_tensor, loss_fn_node=None):
        is_last_stage = self.is_pipeline_first_stage() and phase == 1
        if is_last_stage and self._compute_loss:
            if isinstance(loss_tensor, (tuple, list)):
                assert len(loss_tensor) == 1
                loss_tensor = loss_tensor[0]
            assert isinstance(
                loss_tensor, paddle.Tensor
            ), "Currently, loss_fn should obtain Paddle.Tensor dtype"

            self.loss_tensors.append(loss_tensor)
            self.loss_fn_chunks.append(loss_fn_node)

    def _backward_compute(self, phase: int, enable_zb: bool = False) -> None:
        if self.forward_only:
            return

        acc_id = self.current_b_acc_id[phase]
        self.current_b_acc_id[phase] += 1

        is_last_stage = self.is_pipeline_first_stage() and phase == 1

        WeightGradStore.enabled = enable_zb
        input_grads = None
        with paddle.amp.auto_cast(enable=False):
            if is_last_stage:
                loss = self.loss_tensors[acc_id]
                if self.overlapped_forward_backward:
                    loss_fn_node = self.loss_fn_chunks[acc_id]
                    input_grads = loss_fn_node.backward(scaler=self.scaler)
                    backward_chunk = self.schedule_chunks[phase][acc_id]
                    input_grads = backward_chunk.backward(input_grads)
                    self.loss_fn_chunks[acc_id] = None
                    self.schedule_chunks[phase][acc_id] = None
                else:
                    if self.scaler:
                        paddle.autograd.backward(self.scaler.scale(loss))
                    else:
                        paddle.autograd.backward(loss)
            else:
                outputs, output_grads = self._get_backward_inputs(phase, acc_id)
                if self.overlapped_forward_backward:
                    backward_chunk = self.schedule_chunks[phase][acc_id]
                    input_grads = backward_chunk.backward(output_grads)
                    self.schedule_chunks[phase][acc_id] = None
                else:
                    if len(outputs) > 0:
                        outputs = [t for t in outputs if not t.stop_gradient]
                        paddle.autograd.backward(
                            tensors=outputs,
                            grad_tensors=output_grads,
                        )
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()

        self._store_backward_tensors(phase, acc_id, input_grads=input_grads)

    def _forward_backward_compute(
        self,
        forward_phase: int,
        backward_phase: int,
        micro_datasets=None,
        combine_backward_event_to_wait=None,
        pass_pp_stream=False,
    ) -> None:
        if self.forward_only:
            self._forward_compute(forward_phase, micro_datasets)
            return

        if not self.overlapped_forward_backward:
            self._forward_compute(forward_phase, micro_datasets)
            self._backward_compute(backward_phase)
            return

        # pre-forward
        forward_acc_id = self.current_f_acc_id[forward_phase]
        self.current_f_acc_id[forward_phase] += 1

        forward_inputs = self._get_forward_inputs(
            micro_datasets, forward_phase, forward_acc_id
        )
        forward_labels = self._get_forward_labels(
            micro_datasets, forward_phase, forward_acc_id
        )
        if forward_labels is not None:
            forward_loss_fn_node = self._layers._loss_fn[
                0
            ].build_schedule_node()
            forward_loss_fn_node.labels = forward_labels
            if self.accumulate_steps > 1 and not self._delay_scale_loss:
                forward_loss_fn_node.scale_loss_factor = self.accumulate_steps
        else:
            forward_loss_fn_node = None

        # pre-backward
        backward_acc_id = self.current_b_acc_id[backward_phase]
        self.current_b_acc_id[backward_phase] += 1

        is_last_stage1 = self.is_pipeline_first_stage() and backward_phase == 1
        if is_last_stage1:
            backward_loss_fn_node = self.loss_fn_chunks[backward_acc_id]
            backward_grads = None
        else:
            backward_loss_fn_node = None
            _, backward_grads = self._get_backward_inputs(
                backward_phase, backward_acc_id
            )

        # event_to_wait = deep_ep.get_event_from_custom_stream(paddle.device.current_stream().stream_base)

        # forward & backward
        forward_chunk = self._layers.get_schedule_chunk(chunk_id=forward_phase)
        backward_chunk = self.schedule_chunks[backward_phase][backward_acc_id]
        forward_outputs, forward_loss, backward_input_grads = (
            self._layers.overlapped_forward_backward(
                forward_chunk,
                forward_inputs,
                forward_loss_fn_node,
                backward_chunk,
                backward_loss_fn_node,
                backward_grads,
                self.scaler,
                combine_bw_event_to_wait=combine_backward_event_to_wait,
                pp_stream=(
                    self.pp_group.process_group.get_stream(
                        paddle.framework._current_expected_place_()
                    )
                    if pass_pp_stream
                    else None
                ),
            )
        )
        self.schedule_chunks[backward_phase][backward_acc_id] = None

        # post-forward
        self._store_forward_tensors(
            forward_phase, forward_outputs, forward_chunk
        )
        self._store_forward_loss(
            forward_phase, forward_loss, forward_loss_fn_node
        )

        # post-backward
        self._store_backward_tensors(
            backward_phase, backward_acc_id, input_grads=backward_input_grads
        )

    def _commit_and_wait_comm(
        self, p2p_overlap=False, use_outer_event_wait=False
    ) -> None:
        common_forward_ops_num = (
            len(self.comm_forward_ops)
            if self.comm_forward_ops is not None
            else 0
        )
        common_backward_ops_num = (
            len(self.comm_backward_ops)
            if self.comm_backward_ops is not None
            else 0
        )
        if common_forward_ops_num == 0 and common_backward_ops_num == 0:
            return deep_ep.get_event_from_custom_stream(
                paddle.device.current_stream().stream_base
            )

        use_stream_wait_event = (
            p2p_overlap and self._overlap_p2p_comm and deep_ep is not None
        )

        pp_raw_stream = self.pp_group.process_group.get_stream(
            paddle.framework._current_expected_place_()
        )
        if use_outer_event_wait:
            self.pp_group.process_group.set_outer_wait(True)

        if common_forward_ops_num > 0:
            fwd_reqs = batch_isend_irecv(self.comm_forward_ops)

            if not use_stream_wait_event:
                for req in fwd_reqs:
                    req.wait()

        if use_outer_event_wait:
            self.pp_group.process_group.set_outer_wait(False)

        if use_stream_wait_event:
            forward_event_to_wait = deep_ep.get_event_from_custom_stream(
                pp_raw_stream
            )

        if common_backward_ops_num > 0:
            bwd_reqs = batch_isend_irecv(self.comm_backward_ops)

            if not use_stream_wait_event:
                for req in bwd_reqs:
                    req.wait()

        if use_stream_wait_event:
            forward_event_to_wait.current_stream_wait()

            combine_bw_event_to_wait = deep_ep.get_event_from_custom_stream(
                pp_raw_stream
            )
        else:
            combine_bw_event_to_wait = deep_ep.get_event_from_custom_stream(
                paddle.device.current_stream().stream_base
            )

        self.comm_forward_ops = []
        self.comm_backward_ops = []

        self._free_tensors()

        return combine_bw_event_to_wait

    def _weight_pass(self) -> None:
        if self.forward_only:
            return

        self._commit_and_wait_comm()

        # Assume FIFO
        WeightGradStore.pop()

    def _free_tensors(self) -> None:
        self._release_output(self.to_free)
        self.to_free = []

    def _recv_forward(self, phase: int) -> None:
        if (self.is_pipeline_first_stage() and phase == 0) or (
            self.is_pipeline_last_stage() and phase == 1
        ):
            return

        self.current_recv_f_acc_id[phase] += 1

        tensors = self._p2p_helper.append_irecv(
            self.comm_forward_ops,
            self.prev_rank if phase == 0 else self.next_rank,
            self.pp_group,
            alloc_on_comm_stream=self._overlap_p2p_comm,
        )
        self.input_tensors[phase].append(tensors)

    def _send_forward(self, phase: int) -> None:
        if (self.is_pipeline_first_stage() and phase == 1) or (
            self.is_pipeline_last_stage() and phase == 0
        ):
            return

        acc_id = self.current_send_f_acc_id[phase]
        self.current_send_f_acc_id[phase] += 1
        tensors = self.output_tensors[phase][acc_id]

        self._p2p_helper.append_isend(
            self.comm_forward_ops,
            tensors,
            self.next_rank if phase == 0 else self.prev_rank,
            self.pp_group,
            self.need_broadcast_meta,
        )
        self.need_broadcast_meta = False

        self.to_free.extend(tensors)

    def _recv_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_pipeline_first_stage() and phase == 1) or (
            self.is_pipeline_last_stage() and phase == 0
        ):
            return

        self.current_recv_b_acc_id[phase] += 1
        tensors = self._p2p_helper.append_irecv(
            self.comm_backward_ops,
            self.next_rank if phase == 0 else self.prev_rank,
            self.pp_group,
            alloc_on_comm_stream=self._overlap_p2p_comm,
        )
        self.output_grad_tensors[phase].append(tensors)

    def _send_backward(self, phase: int) -> None:
        if self.forward_only:
            return

        if (self.is_pipeline_first_stage() and phase == 0) or (
            self.is_pipeline_last_stage() and phase == 1
        ):
            return

        acc_id = self.current_send_b_acc_id[phase]
        self.current_send_b_acc_id[phase] += 1
        tensors = self.input_grad_tensors[phase][acc_id]
        self.input_grad_tensors[phase][acc_id] = None

        self._p2p_helper.append_isend(
            self.comm_backward_ops,
            tensors,
            self.prev_rank if phase == 0 else self.next_rank,
            self.pp_group,
        )

    def _forward_pass(
        self,
        phase: int,
        micro_datasets=None,
        recv: bool = True,
        send: bool = True,
    ) -> None:
        if recv:
            self._recv_forward(phase)
        self._commit_and_wait_comm()

        self._forward_compute(phase, micro_datasets)

        if send:
            self._send_forward(phase)

    def _backward_pass(
        self,
        phase: int,
        enable_zb: bool = False,
        recv: bool = True,
        send: bool = True,
    ) -> None:
        if recv:
            self._recv_backward(phase)
        self._commit_and_wait_comm()

        self._backward_compute(phase, enable_zb)

        if send:
            self._send_backward(phase)

    def _forward_backward_pass(
        self,
        forward_phase: int,
        backward_phase: int,
        micro_datasets=None,
        recv0: bool = True,
        first_chunk=False,
        last_chunk=False,
        main_stage=False,
        last_stage_and_first_chunk=False,
    ) -> None:
        if recv0:
            self._recv_forward(forward_phase)
        self._recv_backward(backward_phase)

        need_send_forward = not (
            self.is_pipeline_first_stage() and forward_phase == 1
        ) or (self.is_pipeline_last_stage() and forward_phase == 0)
        need_send_backward = not (
            self.is_pipeline_first_stage() and backward_phase == 0
        ) or (self.is_pipeline_last_stage() and backward_phase == 1)

        use_outer_event_wait = (
            main_stage
            and not first_chunk
            and self._overlap_p2p_comm
            and deep_ep is not None
            and (need_send_forward and need_send_backward)
        )

        pass_pp_stream = (
            main_stage
            and not last_chunk
            and self._overlap_p2p_comm
            and deep_ep is not None
            and (need_send_forward and need_send_backward)
            and (not last_stage_and_first_chunk)
        )

        combine_bw_wait_event = self._commit_and_wait_comm(
            not last_chunk, use_outer_event_wait
        )

        self._forward_backward_compute(
            forward_phase,
            backward_phase,
            micro_datasets,
            combine_backward_event_to_wait=combine_bw_wait_event,
            pass_pp_stream=pass_pp_stream,
        )

        self._send_forward(forward_phase)
        self._send_backward(backward_phase)

    def _wrap_data(self, data, phase):
        """
        for backward compatibility, wrap data to Fake FakeMicroDataset if it is of type list or tuple
        """
        if (not isinstance(data, tuple)) and (not isinstance(data, list)):
            return data

        micro_dataset = FakeMicroDataset(
            data,
            self.is_pipeline_first_stage() and phase == 0,
            self.is_pipeline_first_stage() and phase == 1,
            self.accumulate_steps,
            self.micro_batch_size,
        )
        return micro_dataset

    def _prepare_training(self, data, optimizer, lr_scheduler):
        assert isinstance(
            optimizer, HybridParallelOptimizer
        ), 'optimizer should be HybridParallelOptimizer subclass.'

        assert (
            framework._dygraph_tracer()._has_grad
        ), 'Please enable the generation of gradients.'

        if self.is_pipeline_first_stage():
            assert (
                data is not None
            ), "For the first and the last stage, the data must be set."
        else:
            data = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self._layers.train()
        self.register_sharding_comm_overlap_hook(optimizer)

        return data

    def _broadcast_final_loss(self):
        loss_sum_tensor = paddle.zeros([1], "float32")
        if self.is_pipeline_first_stage():
            assert (
                len(self.loss_tensors) > 0
            ), "train_batch() in last stage should obtain valid loss"
            for loss in self.loss_tensors:
                loss_sum_tensor += loss.detach().astype("float32")
            if self._delay_scale_loss:
                loss_sum_tensor /= self.accumulate_steps

        paddle.distributed.all_reduce(
            loss_sum_tensor, group=self.pp_group, sync_op=True
        )
        return loss_sum_tensor

    def forward_backward_pipeline(
        self,
        data,
        scaler,
        forward_only=False,
        compute_loss=True,
    ):
        self.scaler = scaler

        rank = self.group_rank
        num_ranks = self.num_ranks
        assert (
            self.accumulate_steps > 0 and self.accumulate_steps >= num_ranks * 2
        ), f"{self.accumulate_steps=}, {num_ranks=}"
        self.forward_only = forward_only

        self._reset_states()

        # NOTE(zhangyuqin1998): Tensors to be sent or received must have a
        # consistent shape and data type throughout the entire pipeline. We
        # broadcast the meta info in the first forward of the first rank.
        self._p2p_helper.recv_meta_from_head(self.pp_group, self.need_recv_meta)
        self.need_recv_meta = False

        micro_dataset_phase0 = self._wrap_data(data, 0)
        micro_dataset_phase1 = self._wrap_data(data, 1)
        micro_datasets = [micro_dataset_phase0, micro_dataset_phase1]

        # Step 1: nF0
        step_1 = (num_ranks - rank - 1) * 2
        for i in range(step_1):
            self._forward_pass(0, micro_datasets)

        # Step 2: nF0F1
        step_2 = rank + 1
        self._recv_forward(0)
        for i in range(step_2):
            self._forward_pass(0, micro_datasets, recv=False, send=False)
            self._recv_forward(0)
            self._forward_pass(
                1,
                micro_datasets,
                send=(not self.is_pipeline_last_stage()) or (i < step_2 - 1),
            )
            self._send_forward(0)

        # Step 3: nB1W1F1 (Use zero bubble)
        step_3 = num_ranks - rank - 1
        for i in range(step_3):
            self._backward_pass(1, enable_zb=True)
            self._recv_forward(1)
            self._weight_pass()
            self._forward_pass(1, micro_datasets, recv=False)

        # Step 4 (Main step): nF0B1F1B0
        step_4 = self.accumulate_steps - num_ranks * 2 + rank + 1
        have_step5 = num_ranks - rank - 1 > 0
        # Update code to support send/recv overlap
        # Only support send/recv overlap in MainStep
        for i in range(step_4):
            is_last_chunk = i + 1 == step_4
            if i == 0:
                if self.is_pipeline_last_stage():
                    # NOTE: We don't overlap these two passes to further reduce bubble size.
                    self._forward_pass(
                        0, micro_datasets, recv=False, send=False
                    )
                    self._send_forward(1)
                    self._backward_pass(1, send=False)
                    self._send_forward(0)
                    self._send_backward(1)

                    self._forward_backward_pass(
                        1,
                        0,
                        micro_datasets,
                        first_chunk=True,
                        last_chunk=is_last_chunk,
                        main_stage=True,
                    )
                else:
                    self._forward_backward_pass(
                        0,
                        1,
                        micro_datasets,
                        recv0=False,
                        first_chunk=True,
                        main_stage=True,
                    )

                    self._forward_backward_pass(
                        1,
                        0,
                        micro_datasets,
                        last_chunk=is_last_chunk,
                        main_stage=True,
                    )
            else:

                self._forward_backward_pass(
                    0,
                    1,
                    micro_datasets,
                    main_stage=True,
                    last_stage_and_first_chunk=self.is_pipeline_last_stage(),
                )
                self._forward_backward_pass(
                    1,
                    0,
                    micro_datasets,
                    last_chunk=is_last_chunk,
                    main_stage=True,
                )

        # Step 5: nB1F1B0
        step_5 = num_ranks - rank - 1
        for i in range(step_5):
            self._backward_pass(1)
            self._forward_backward_pass(1, 0, micro_datasets)

        # Step 6: nB1B0 (The second half of the passes use zero bubble)
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True
            self._backward_pass(1, enable_zb=enable_zb)
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True
            self._backward_pass(0, enable_zb=enable_zb)

        # Step 7: nWB0 (Use zero bubble)
        step_7 = num_ranks - rank - 1
        for i in range(step_7):
            self._weight_pass()
            self._backward_pass(0, enable_zb=True)

        # Step 8: nW
        step_8 = rank + 1
        for i in range(step_8):
            self._weight_pass()
        assert WeightGradStore.funcs_queue.empty()

        self._commit_and_wait_comm()

        self._layers.allreduce_shared_weight_gradients()

        with paddle.amp.auto_cast(enable=False):
            train_loss = self._broadcast_final_loss()

        self._reset_states()
        return train_loss

    def train_batch(
        self,
        data,
        optimizer,
        lr_scheduler=None,
        scaler=None,
    ):
        data = self._prepare_training(data, optimizer, lr_scheduler)

        train_loss = self.forward_backward_pipeline(data, scaler)

        # optimizer
        with paddle.amp.auto_cast(enable=False):
            self._optimizer_step()

        return train_loss

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

import itertools

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F


def _get_comm_group_by_dim(mesh, dim):
    dim_names = mesh.dim_names
    assert dim in dim_names, f"dim '{dim}' not in mesh.dim_names {dim_names}"

    shape = mesh.shape
    dim_idx = dim_names.index(dim)
    ids = mesh.process_ids

    def nest(flat, shape):
        if not shape:
            return flat[0]
        step = int(len(flat) // shape[0])
        return [
            nest(flat[i * step : (i + 1) * step], shape[1:])
            for i in range(shape[0])
        ]

    mesh_nd = nest(ids, shape)

    other_axes = [i for i in range(len(shape)) if i != dim_idx]
    other_ranges = [range(shape[i]) for i in other_axes]

    comm_groups = []
    for index in itertools.product(*other_ranges):
        group = []
        for i in range(shape[dim_idx]):
            idx = list(index)
            idx.insert(dim_idx, i)
            val = mesh_nd
            for j in idx:
                val = val[j]
            group.append(val)
        comm_groups.append(group)

    return comm_groups


def _get_conv_tp_group(x_mesh, x_placements, data_format):
    if data_format == "NCHW":
        shard_axis = 3
    else:
        shard_axis = 2

    axis_name = None
    for i, placement in enumerate(x_placements):
        if placement == dist.Shard(shard_axis):
            axis_name = x_mesh.dim_names[i]
            break

    if not axis_name:
        raise ValueError(
            f"Input tensor placements {x_placements} do not contain a Shard on W axis:{shard_axis}."
        )

    tp_groups = _get_comm_group_by_dim(x_mesh, axis_name)
    rank = dist.get_rank()

    for group in tp_groups:
        if rank in group:
            return axis_name, group

    raise RuntimeError(
        f"Rank {rank} not found in any tensor parallel group for mesh {x_mesh}."
    )


def _ring_send_recv_construct(
    local_input_tensor,
    halo_width_to_receive_from_left,
    halo_width_to_receive_from_right,
    left_neighbor_rank,
    right_neighbor_rank,
    current_rank,
    conv_tp_group,
    data_format,
):
    if len(conv_tp_group) == 1:
        return local_input_tensor

    if not (
        len(local_input_tensor.shape) == 4
    ):  # Assuming 4D tensors like NCHW/NHWC
        raise ValueError(
            f"Input tensor is expected to be 4D for NCHW/NHWC formats, "
            f"but got {len(local_input_tensor.shape)}D."
        )

    if data_format == "NCHW":
        width_dim_idx = 3
    elif data_format == "NHWC":
        width_dim_idx = 2
    else:
        raise ValueError(
            f"Unsupported data_format: {data_format}. Must be 'NCHW' or 'NHWC'."
        )

    # Segment to send to the right neighbor (right_neighbor_rank)
    slices_for_send_right = [slice(None)] * 4
    slices_for_send_right[width_dim_idx] = slice(
        -halo_width_to_receive_from_left, None
    )
    segment_to_send_right = local_input_tensor[
        tuple(slices_for_send_right)
    ].contiguous()

    # Segment to send to the left neighbor (left_neighbor_rank)
    slices_for_send_left = [slice(None)] * 4
    slices_for_send_left[width_dim_idx] = slice(
        None, halo_width_to_receive_from_right
    )
    segment_to_send_left = local_input_tensor[
        tuple(slices_for_send_left)
    ].contiguous()

    buffer_for_halo_from_right = paddle.zeros_like(segment_to_send_left)
    buffer_for_halo_from_left = paddle.zeros_like(segment_to_send_right)

    op_isend_to_right = dist.P2POp(
        dist.isend, segment_to_send_right, right_neighbor_rank
    )
    op_isend_to_left = dist.P2POp(
        dist.isend, segment_to_send_left, left_neighbor_rank
    )

    op_irecv_from_right = dist.P2POp(
        dist.irecv, buffer_for_halo_from_right, right_neighbor_rank
    )
    op_irecv_from_left = dist.P2POp(
        dist.irecv, buffer_for_halo_from_left, left_neighbor_rank
    )

    p2p_requests = dist.batch_isend_irecv(
        [
            op_isend_to_right,
            op_isend_to_left,
            op_irecv_from_left,
            op_irecv_from_right,
        ]
    )
    for req in p2p_requests:
        req.wait()

    # Concatenate received halo regions with the local tensor
    if current_rank == conv_tp_group[0]:
        # First rank: original tensor || halo_from_right
        reconstructed_tensor = paddle.concat(
            [local_input_tensor, buffer_for_halo_from_right], axis=width_dim_idx
        )
    elif current_rank == conv_tp_group[-1]:
        # Last rank: halo_from_left || original tensor
        reconstructed_tensor = paddle.concat(
            [buffer_for_halo_from_left, local_input_tensor], axis=width_dim_idx
        )
    else:
        # Middle ranks: halo_from_left || original tensor || halo_from_right
        reconstructed_tensor = paddle.concat(
            [
                buffer_for_halo_from_left,
                local_input_tensor,
                buffer_for_halo_from_right,
            ],
            axis=width_dim_idx,
        )

    return reconstructed_tensor


def _ring_send_recv_aggregate(
    local_gradient_tensor,
    halo_width_send_left,
    halo_width_send_right,
    left_neighbor_rank,
    right_neighbor_rank,
    current_process_rank,
    conv_tp_group,
    data_format,
):
    if len(conv_tp_group) == 1:
        return local_gradient_tensor

    if data_format == "NCHW":
        width_dim_idx = 3
    elif data_format == "NHWC":
        width_dim_idx = 2
    else:
        raise ValueError(
            f"Unsupported data_format: {data_format}. Must be 'NCHW' or 'NHWC'."
        )

    # Prepare gradient segments to send
    slices_for_send_right = [slice(None)] * 4
    slices_for_send_right[width_dim_idx] = slice(
        -halo_width_send_right, None
    )  # Send the rightmost part
    segment_to_send_right = local_gradient_tensor[
        tuple(slices_for_send_right)
    ].contiguous()

    slices_for_send_left = [slice(None)] * 4
    slices_for_send_left[width_dim_idx] = slice(
        None, halo_width_send_left
    )  # Send the leftmost part
    segment_to_send_left = local_gradient_tensor[
        tuple(slices_for_send_left)
    ].contiguous()

    # Buffers for receiving gradients
    buffer_for_gradient_from_left = paddle.zeros_like(segment_to_send_right)
    buffer_for_gradient_from_right = paddle.zeros_like(segment_to_send_left)

    op_isend_to_right = dist.P2POp(
        dist.isend, segment_to_send_right, right_neighbor_rank
    )
    op_isend_to_left = dist.P2POp(
        dist.isend, segment_to_send_left, left_neighbor_rank
    )
    op_irecv_from_right = dist.P2POp(
        dist.irecv, buffer_for_gradient_from_right, right_neighbor_rank
    )
    op_irecv_from_left = dist.P2POp(
        dist.irecv, buffer_for_gradient_from_left, left_neighbor_rank
    )

    p2p_requests = dist.batch_isend_irecv(
        [
            op_isend_to_right,
            op_isend_to_left,
            op_irecv_from_left,
            op_irecv_from_right,
        ]
    )
    for req in p2p_requests:
        req.wait()

    processed_gradient_tensor = local_gradient_tensor
    # Crop local tensor and aggregate received gradients
    if current_process_rank == conv_tp_group[0]:
        # Crop the part sent to the right neighbor
        crop_slices = [slice(None)] * 4
        crop_slices[width_dim_idx] = slice(None, -halo_width_send_right)
        processed_gradient_tensor = processed_gradient_tensor[
            tuple(crop_slices)
        ]

        # Aggregate gradient received from the right neighbor
        # This is added to the new rightmost part of the processed_gradient_tensor
        agg_slices = [slice(None)] * 4
        agg_slices[width_dim_idx] = slice(-halo_width_send_left, None)

        target_slice = processed_gradient_tensor[tuple(agg_slices)]
        target_slice.add_(buffer_for_gradient_from_right)

    elif current_process_rank == conv_tp_group[-1]:
        # Crop the part sent to the left neighbor
        crop_slices = [slice(None)] * 4
        crop_slices[width_dim_idx] = slice(halo_width_send_left, None)
        processed_gradient_tensor = processed_gradient_tensor[
            tuple(crop_slices)
        ]

        # Aggregate gradient received from the left neighbor
        agg_slices = [slice(None)] * 4
        agg_slices[width_dim_idx] = slice(None, halo_width_send_right)

        target_slice = processed_gradient_tensor[tuple(agg_slices)]
        target_slice.add_(buffer_for_gradient_from_left)

    else:
        # Crop parts sent to both left and right neighbors
        crop_slices = [slice(None)] * 4
        crop_slices[width_dim_idx] = slice(
            halo_width_send_left, -halo_width_send_right
        )
        processed_gradient_tensor = processed_gradient_tensor[
            tuple(crop_slices)
        ]

        # Aggregate gradient received from the right neighbor
        agg_slices_right_edge = [slice(None)] * 4
        agg_slices_right_edge[width_dim_idx] = slice(
            -halo_width_send_left, None
        )
        target_slice_right = processed_gradient_tensor[
            tuple(agg_slices_right_edge)
        ]
        target_slice_right.add_(buffer_for_gradient_from_right)

        # Aggregate gradient received from the left neighbor
        agg_slices_left_edge = [slice(None)] * 4
        agg_slices_left_edge[width_dim_idx] = slice(None, halo_width_send_right)
        target_slice_left = processed_gradient_tensor[
            tuple(agg_slices_left_edge)
        ]
        target_slice_left.add_(buffer_for_gradient_from_left)

    return processed_gradient_tensor


class RingConv2d(paddle.autograd.PyLayer):
    @staticmethod
    def _is_supported(
        input_size, kernel_size, stride, padding, dilation, data_format="NCHW"
    ):
        idx_w_input = -1
        idx_w_kernel = -1

        if data_format == "NCHW":
            # input_size: (N, C, H, W)
            # kernel_size: (OutChannels, InChannels/Groups, KernelH, KernelW)
            idx_w_input = 3
            idx_w_kernel = 3
        elif data_format == "NHWC":
            # input_size: (N, H, W, C)
            # kernel_size: (OutChannels, InChannels/Groups, KernelH, KernelW)
            idx_w_input = 2
            idx_w_kernel = 3
        else:
            raise ValueError(
                f"Unsupported data_format '{data_format}'. Expected 'NCHW' or 'NHWC'."
            )

        dilation_w = dilation[1]
        padding_w = padding[1]
        stride_w = stride[1]

        input_w = input_size[idx_w_input]
        kernel_w = kernel_size[idx_w_kernel]

        if dilation_w != 1:
            # RingConv2d only supports dilation=1.
            # Larger dilation would require enlarged halo regions and more complex communication.
            raise RuntimeError(
                f"Only dilation=1 on the W-dimension is supported for tensor-parallel convolution. "
                f"Got dilation_w={dilation_w} (data_format='{data_format}')."
            )

        if padding_w == 0:
            # To avoid halo exchange when padding=0, we require:
            # - input_w must be divisible by stride_w, so partitions align evenly across ranks.
            # - stride_w == kernel_w, so each kernel operates on disjoint local regions.
            if input_w % stride_w != 0:
                raise RuntimeError(
                    f"When padding_w=0, input_W={input_w} must be divisible by stride_W={stride_w} "
                    f"for tensor-parallel convolution (data_format='{data_format}')."
                )
            if stride_w != kernel_w:
                raise RuntimeError(
                    f"When padding_w=0, stride_W={stride_w} must equal kernel_W={kernel_w} "
                    f"to avoid halo exchange (data_format='{data_format}')."
                )

        else:
            # When padding > 0, halo exchange is needed.
            # To simplify halo logic, we require:
            # - stride_w == 1: ensures each output element is computed from overlapping input,
            #   and no input region is skipped, simplifying halo construction.
            # - kernel_w // 2 <= input_w: prevents the kernel from exceeding local input.
            if stride_w != 1:
                raise RuntimeError(
                    f"When padding_w={padding_w}, stride_W must be 1 for tensor-parallel convolution. "
                    f"Got stride_W={stride_w} (data_format='{data_format}')."
                )
            if kernel_w // 2 > input_w:
                raise RuntimeError(
                    f"Half of kernel_W ({kernel_w // 2}) must not exceed input_W={input_w} "
                    f"to ensure halo region fits (data_format='{data_format}')."
                )

        return True

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        stride=1,
        padding=0,
        padding_algorithm=None,
        dilation=1,
        groups=1,
        data_format="NCHW",
        channel_dim=1,
    ):
        rank = dist.get_rank()

        assert RingConv2d._is_supported(
            x.shape, weight.shape, stride, padding, dilation, data_format
        )
        assert x.is_dist(), "Input tensor `x` must be a distributed tensor."

        if not weight.is_dist():
            weight_placements = [
                dist.Replicate() for _ in range(len(x.placements))
            ]
            weight = dist.auto_parallel.api.dtensor_from_local(
                weight, x.process_mesh, weight_placements
            )

        if bias is not None and not bias.is_dist():
            bias_placements = [
                dist.Replicate() for _ in range(len(x.placements))
            ]
            bias = dist.auto_parallel.api.dtensor_from_local(
                bias, x.process_mesh, bias_placements
            )

        ctx.save_for_backward(x, weight, bias)

        x_mesh = x.process_mesh
        x_placements = x.placements
        x = dist.auto_parallel.api.dtensor_to_local(x, x_mesh, x_placements)

        weight = dist.auto_parallel.api.dtensor_to_local(
            weight, weight.process_mesh, weight.placements
        )
        if bias is not None:
            bias = dist.auto_parallel.api.dtensor_to_local(
                bias, bias.process_mesh, bias.placements
            )

        ctx.attrs = (
            stride,
            padding,
            padding_algorithm,
            dilation,
            groups,
            data_format,
        )

        mesh_axis_name, conv_tp_group = _get_conv_tp_group(
            x_mesh, x_placements, data_format
        )
        if padding[1] == 0 or len(conv_tp_group) <= 1:
            final_local_results = paddle._C_ops.conv2d(
                x,
                weight,
                stride,
                padding,
                padding_algorithm,
                dilation,
                groups,
                data_format,
            )
        else:
            # step 0: calculate the required overlap (halo) pixels for the input tensor
            if data_format == "NCHW":
                kernel_width_dim_idx = 3
                output_width_dim_idx = 3
            elif data_format == "NHWC":
                kernel_width_dim_idx = 3
                output_width_dim_idx = 2
            else:
                raise ValueError(
                    f"Unsupported data_format: {data_format}. Must be 'NCHW' or 'NHWC'."
                )

            kernel_width = weight.shape[kernel_width_dim_idx]
            kernel_total_halo_span = kernel_width - 1
            left_halo_width = kernel_total_halo_span // 2
            right_halo_width = kernel_total_halo_span - left_halo_width
            assert left_halo_width + right_halo_width == kernel_total_halo_span

            ctx.mesh_axis_name = mesh_axis_name
            rank_idx = conv_tp_group.index(rank)
            next_rank = conv_tp_group[(rank_idx + 1) % len(conv_tp_group)]
            prev_rank = conv_tp_group[(rank_idx - 1) % len(conv_tp_group)]

            # step 1: reconstruct the local input tensor including halo regions via ring communication
            # `x` is updated here, now including halo data received from neighboring ranks.
            x = _ring_send_recv_construct(
                x,
                left_halo_width,
                right_halo_width,
                prev_rank,
                next_rank,
                rank,
                conv_tp_group,
                data_format,
            )

            # step 2: feed the reconstructed local input tensor to the actual computation (op_call)
            local_results_with_halo = paddle._C_ops.conv2d(
                x,
                weight,
                stride,
                padding,
                padding_algorithm,
                dilation,
                groups,
                data_format,
            )

            # step 3: remove extra output portions from the results, generated from processing halo regions
            # `padding[1]` (from outer scope) is assumed here to be the width of the halo/overlap
            # that needs to be trimmed from each side of the output.
            output_halo_trim_width = padding[1]
            width_before_trimming = local_results_with_halo.shape[
                output_width_dim_idx
            ]

            if data_format == "NCHW":
                if rank == conv_tp_group[0]:
                    final_local_results = local_results_with_halo[
                        :,
                        :,
                        :,
                        : width_before_trimming - output_halo_trim_width,
                    ]
                elif rank == conv_tp_group[-1]:
                    final_local_results = local_results_with_halo[
                        :, :, :, output_halo_trim_width:
                    ]
                else:
                    final_local_results = local_results_with_halo[
                        :,
                        :,
                        :,
                        output_halo_trim_width : width_before_trimming
                        - output_halo_trim_width,
                    ]
            else:
                if rank == conv_tp_group[0]:
                    final_local_results = local_results_with_halo[
                        :,
                        :,
                        : width_before_trimming - output_halo_trim_width,
                        :,
                    ]
                elif rank == conv_tp_group[-1]:
                    final_local_results = local_results_with_halo[
                        :, :, output_halo_trim_width:, :
                    ]
                else:
                    final_local_results = local_results_with_halo[
                        :,
                        :,
                        output_halo_trim_width : width_before_trimming
                        - output_halo_trim_width,
                        :,
                    ]

            ctx.left_halo_width = left_halo_width
            ctx.right_halo_width = right_halo_width
            ctx.output_halo_trim_width = output_halo_trim_width
            ctx.output_width_dim_idx = output_width_dim_idx

        final_local_results = dist.auto_parallel.api.dtensor_from_local(
            final_local_results, x_mesh, x_placements
        )

        return final_local_results

    @staticmethod
    def backward(ctx, grad_out):
        current_rank = dist.get_rank()
        x, weight, bias = ctx.saved_tensor()

        x_stop_gradient = x.stop_gradient
        weight_stop_gradient = weight.stop_gradient
        bias_stop_gradient = bias.stop_gradient if bias is not None else True

        x_mesh = x.process_mesh
        x_placements = x.placements
        x = dist.auto_parallel.api.dtensor_to_local(x, x_mesh, x_placements)

        weight_mesh = weight.process_mesh
        weight_placements = weight.placements
        weight = dist.auto_parallel.api.dtensor_to_local(
            weight, weight_mesh, weight_placements
        )

        grad_out = dist.auto_parallel.api.dtensor_to_local(
            grad_out, grad_out.process_mesh, grad_out.placements
        )

        if bias is not None:
            bias_mesh = bias.process_mesh
            bias_placements = bias.placements
            bias = dist.auto_parallel.api.dtensor_to_local(
                bias, bias_mesh, bias_placements
            )

        conv_attrs = ctx.attrs
        data_format = conv_attrs[-1]
        padding = conv_attrs[1]

        grad_x = None
        grad_weight = None
        grad_bias = None

        _, conv_tp_group = _get_conv_tp_group(x_mesh, x_placements, data_format)

        if padding[1] == 0 or len(conv_tp_group) <= 1:
            grad_x, grad_weight = paddle._C_ops.conv2d_grad(
                x, weight, grad_out, *conv_attrs
            )
        else:
            rank_idx = conv_tp_group.index(current_rank)
            next_rank = conv_tp_group[(rank_idx + 1) % len(conv_tp_group)]
            prev_rank = conv_tp_group[(rank_idx - 1) % len(conv_tp_group)]

            left_halo_width = ctx.left_halo_width
            right_halo_width = ctx.right_halo_width
            output_halo_trim_width = ctx.output_halo_trim_width
            output_width_dim_idx = ctx.output_width_dim_idx

            # Step 1: Reconstruct `in_tensor_augmented` (original input to local conv in forward)
            in_tensor_augmented = _ring_send_recv_construct(
                x,
                left_halo_width,
                right_halo_width,
                prev_rank,
                next_rank,
                current_rank,
                conv_tp_group,
                data_format,
            )

            # Step 2: Pad `grad_out` to match the output shape of conv on augmented input
            padding_w = padding[1]
            if data_format == "NCHW":
                if current_rank == conv_tp_group[0]:
                    padding_list = [0, padding_w]
                elif current_rank == conv_tp_group[-1]:
                    padding_list = [padding_w, 0]
                else:
                    padding_list = [padding_w, padding_w]
            else:
                if current_rank == conv_tp_group[0]:
                    padding_list = [0, padding_w, 0, 0]
                elif current_rank == conv_tp_group[-1]:
                    padding_list = [padding_w, 0, 0, 0]
                else:
                    padding_list = [padding_w, padding_w, 0, 0]

            grad_out_padded = F.pad(
                grad_out,
                padding_list,
                mode="constant",
                value=0.0,
                data_format=data_format,
            )

            # Step 3: Local backward computation using augmented/padded tensors
            # `padding` here is the original conv padding from forward.
            grad_x_augmented, grad_weight = paddle._C_ops.conv2d_grad(
                in_tensor_augmented, weight, grad_out_padded, *conv_attrs
            )

            # Step 4: Aggregate "halo" regions for grad_input
            if not x_stop_gradient:
                grad_x = _ring_send_recv_aggregate(
                    grad_x_augmented,
                    left_halo_width,
                    right_halo_width,
                    prev_rank,
                    next_rank,
                    current_rank,
                    conv_tp_group,
                    data_format,
                )

        if bias is not None:
            sum_axes = [0, 2, 3] if data_format == "NCHW" else [0, 1, 2]
            grad_bias = paddle.sum(grad_out, axis=sum_axes, keepdim=True)
            grad_bias = grad_bias.reshape(bias.shape)

        if grad_x is not None:
            grad_x = dist.auto_parallel.api.dtensor_from_local(
                grad_x, x_mesh, x_placements
            )

        # Note(luchang): With input X sharded along tp_axis_name and weight W replicated,
        # the locally computed grad_weight is only a partial sum for the full dL/dW,
        # as dL/dW depends on contributions from all input shards.
        # Aggregation across TP ranks is therefore necessary. Partial(ReduceSum)
        # declares this averaging intent, and reshard to Replicate() executes
        # the AllReduce-average, making the correct averaged grad_weight available
        # and replicated on all TP ranks.
        tp_axis_name, _ = _get_conv_tp_group(x_mesh, x_placements, data_format)
        for idx, axis_name in enumerate(weight_mesh.dim_names):
            if axis_name == tp_axis_name:
                weight_placements[idx] = dist.Partial(dist.ReduceType.kRedSum)
                if bias is not None:
                    bias_placements[idx] = dist.Partial(dist.ReduceType.kRedSum)

        grad_weight = dist.auto_parallel.api.dtensor_from_local(
            grad_weight, weight_mesh, weight_placements
        )
        grad_weight = dist.reshard(
            grad_weight,
            weight_mesh,
            [dist.Replicate() for _ in range(len(weight_placements))],
        )

        if bias is not None:
            grad_bias = dist.auto_parallel.api.dtensor_from_local(
                grad_bias, bias_mesh, bias_placements
            )
        grad_weight = dist.reshard(
            grad_weight,
            weight_mesh,
            [dist.Replicate() for _ in range(len(weight_placements))],
        )

        if x_stop_gradient:
            grad_x = None
        if weight_stop_gradient:
            grad_weight = None
        if bias_stop_gradient:
            grad_bias = None

        if bias is not None:
            return grad_x, grad_weight, grad_bias

        return grad_x, grad_weight

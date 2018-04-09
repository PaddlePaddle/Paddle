/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "hip/hip_runtime.h"
#include <type_traits>
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/gru_compute.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpResetOutput, bool is_batch, typename T>
__global__ void KeGruForwardResetOutput(OpResetOutput op_reset_output,
                                        T *gate_value, T *reset_output_value,
                                        T *prev_output_value, int frame_size,
                                        int batch_size,
                                        ActivationType active_gate) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;

  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    reset_output_value += batch_idx * frame_size;
  }

  T r_prev_out = 0;
  T r_value_reset_output;
  T r_value_update_gate = gate_value[frame_idx + frame_size * 0];
  T r_value_reset_gate = gate_value[frame_idx + frame_size * 1];

  if (prev_output_value) {
    if (is_batch) prev_output_value += batch_idx * frame_size;
    r_prev_out = prev_output_value[frame_idx];
  }

  op_reset_output(&r_value_update_gate, &r_value_reset_gate, &r_prev_out,
                  &r_value_reset_output, active_gate);

  gate_value[frame_idx + frame_size * 0] = r_value_update_gate;
  gate_value[frame_idx + frame_size * 1] = r_value_reset_gate;
  reset_output_value[frame_idx] = r_value_reset_output;
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpFinalOutput, bool is_batch, typename T>
__global__ void KeGruForwardFinalOutput(OpFinalOutput op_final_output,
                                        T *gate_value, T *prev_output_value,
                                        T *output_value, int frame_size,
                                        int batch_size,
                                        ActivationType active_node) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;
  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    output_value += batch_idx * frame_size;
  }

  T r_output;
  T r_prev_out = 0;
  T r_value_update_gate = gate_value[frame_idx + frame_size * 0];
  T r_value_frame_state = gate_value[frame_idx + frame_size * 2];

  if (prev_output_value) {
    if (is_batch) prev_output_value += batch_idx * frame_size;
    r_prev_out = prev_output_value[frame_idx];
  }

  op_final_output(&r_value_update_gate, &r_value_frame_state, &r_prev_out,
                  &r_output, active_node);

  gate_value[frame_idx + frame_size * 2] = r_value_frame_state;
  output_value[frame_idx] = r_output;
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpStateGrad, bool is_batch, typename T>
__global__ void KeGruBackwardStateGrad(OpStateGrad op_state_grad, T *gate_value,
                                       T *gate_grad, T *prev_out_value,
                                       T *prev_out_grad, T *output_grad,
                                       int frame_size, int batch_size,
                                       ActivationType active_node) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;
  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    gate_grad += batch_idx * 3 * frame_size;
    output_grad += batch_idx * frame_size;
  }

  T r_update_gate_grad;
  T r_frame_state_grad;
  T r_prev_out_value = 0;
  T r_prev_out_grad = 0;
  T r_update_gate_value = gate_value[frame_idx + frame_size * 0];
  T r_frame_state_value = gate_value[frame_idx + frame_size * 2];
  T r_out_grad = output_grad[frame_idx];

  if (prev_out_value && prev_out_grad) {
    if (is_batch) prev_out_value += batch_idx * frame_size;
    r_prev_out_value = prev_out_value[frame_idx];

    if (is_batch) prev_out_grad += batch_idx * frame_size;
    r_prev_out_grad = prev_out_grad[frame_idx];
  }

  op_state_grad(&r_update_gate_value, &r_update_gate_grad, &r_frame_state_value,
                &r_frame_state_grad, &r_prev_out_value, &r_prev_out_grad,
                &r_out_grad, active_node);

  gate_grad[frame_idx + frame_size * 0] = r_update_gate_grad;
  gate_grad[frame_idx + frame_size * 2] = r_frame_state_grad;
  if (prev_out_grad) {
    prev_out_grad[frame_idx] = r_prev_out_grad;
  }
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpResetGrad, bool is_batch, typename T>
__global__ void KeGruBackwardResetGrad(OpResetGrad op_reset_grad, T *gate_value,
                                       T *gate_grad, T *prev_out_value,
                                       T *prev_out_grad, T *reset_output_grad,
                                       int frame_size, int batch_size,
                                       ActivationType active_gate) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;
  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    gate_grad += batch_idx * 3 * frame_size;
    reset_output_grad += batch_idx * frame_size;
  }

  T r_reset_gate_grad;
  T r_prev_out_value = 0;
  T r_prev_out_grad = 0;
  T r_reset_output_grad = 0;
  T r_update_gate_value = gate_value[frame_idx + frame_size * 0];
  T r_update_gate_grad = gate_grad[frame_idx + frame_size * 0];
  T r_reset_gate_value = gate_value[frame_idx + frame_size * 1];

  if (prev_out_value && prev_out_grad) {
    if (is_batch) prev_out_value += batch_idx * frame_size;
    if (is_batch) prev_out_grad += batch_idx * frame_size;
    r_prev_out_value = prev_out_value[frame_idx];
    r_prev_out_grad = prev_out_grad[frame_idx];
    r_reset_output_grad = reset_output_grad[frame_idx];
  }

  op_reset_grad(&r_update_gate_value, &r_update_gate_grad, &r_reset_gate_value,
                &r_reset_gate_grad, &r_prev_out_value, &r_prev_out_grad,
                &r_reset_output_grad, active_gate);

  gate_grad[frame_idx + frame_size * 0] = r_update_gate_grad;
  gate_grad[frame_idx + frame_size * 1] = r_reset_gate_grad;
  if (prev_out_grad) {
    prev_out_grad[frame_idx] = r_prev_out_grad;
  }
}
}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle

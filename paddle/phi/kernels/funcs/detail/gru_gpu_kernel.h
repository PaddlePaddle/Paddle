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
#include <type_traits>

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/detail/activation_functions.h"
#include "paddle/phi/kernels/funcs/gru_compute.h"

namespace phi {
namespace funcs {
namespace detail {

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpResetOutput, bool is_batch, typename T>
__global__ void KeGruForwardResetOutput(OpResetOutput op_reset_output,
                                        T *gate_value,
                                        T *reset_output_value,
                                        const T *prev_output_value,
                                        int frame_size,
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

  op_reset_output(&r_value_update_gate,
                  &r_value_reset_gate,
                  &r_prev_out,
                  &r_value_reset_output,
                  active_gate);

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
                                        T *gate_value,
                                        const T *prev_output_value,
                                        T *output_value,
                                        int frame_size,
                                        int batch_size,
                                        ActivationType active_node,
                                        bool origin_mode) {
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

  op_final_output(&r_value_update_gate,
                  &r_value_frame_state,
                  &r_prev_out,
                  &r_output,
                  active_node,
                  origin_mode);

  gate_value[frame_idx + frame_size * 2] = r_value_frame_state;
  output_value[frame_idx] = r_output;
}

/*
 * threads(tile_size, 1)
 * grid(frame_blocks, 1)
 */
template <class T, int Tiled_size>
__global__ void KeFastCollectiveGruGate(T *gate_value,
                                        const T *prev_output_value,
                                        const T *gate_weight,
                                        T *reset_output,
                                        int frame_size,
                                        ActivationType active_node) {
  T xt_0 = 0.0f;
  T a0 = 0.0f;
  T c0 = 0.0f;
  T b0[Tiled_size];

  int COL = blockIdx.x * blockDim.x + threadIdx.x;
  int Tiled_mask = ((1 << Tiled_size) - 1);
  // Tiled  matrix multiply using register shift, faster than sm.
  if (prev_output_value) {
    for (int k = 0; k < (((frame_size - 1) / Tiled_size) + 1); ++k) {
      a0 = 0;
      if ((threadIdx.x + k * Tiled_size) < frame_size) {
        a0 = prev_output_value[threadIdx.x + (k * Tiled_size)];
      }
      for (int i = 0; i < Tiled_size; i++) {
        if (COL < frame_size * 2 && (i + k * Tiled_size) < frame_size) {
          b0[i] = gate_weight[(i + k * Tiled_size) * frame_size * 2 + COL];
        }
      }

      for (int i = 0; i < Tiled_size; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        c0 = c0 + __shfl_sync(Tiled_mask, a0, i, Tiled_size) * b0[i];
#else
        c0 = c0 + __shfl(a0, i, Tiled_size) * b0[i];
#endif
      }
    }
  }

  __syncthreads();

  if (COL < frame_size * 2) {
    xt_0 = gate_value[COL];
    c0 += xt_0;
    c0 = forward::activation(c0, active_node);
    gate_value[COL] = c0;
    if (frame_size <= COL && COL < frame_size * 2) {
      T htp_0 = 0.0;
      if (prev_output_value) {
        htp_0 = prev_output_value[COL - frame_size];
      }
      reset_output[COL - frame_size] = c0 * htp_0;
    } else if (COL < frame_size) {
      gate_value[COL] = c0;
    }
  }
}

/*
 * threads(tile_size, 1)
 * grid(frame_blocks, 1)
 */
template <class T, int Tiled_size>
__global__ void KeFastCollectiveGruOut(const T *gate_weight,
                                       const T *prev_out_value,
                                       T *output_value,
                                       T *gate_value,
                                       T *reset_value,
                                       int frame_size,
                                       ActivationType act_node,
                                       bool origin_mode) {
  int COL = blockIdx.x * blockDim.x + threadIdx.x;

  T a0 = 0.0f;
  T b0[Tiled_size];
  T c0 = 0.0f;

  int Tiled_mask = ((1 << Tiled_size) - 1);
  //- Tiled  matrix multiply with register shift
  if (prev_out_value) {
    for (int k = 0; k < (((frame_size - 1) / Tiled_size) + 1); ++k) {
      a0 = 0;
      if ((threadIdx.x + k * Tiled_size) < frame_size) {
        a0 = reset_value[threadIdx.x + (k * Tiled_size)];
      }
      for (int i = 0; i < Tiled_size; i++) {
        if (COL < frame_size && (i + k * Tiled_size) < frame_size) {
          b0[i] = gate_weight[(i + k * Tiled_size) * frame_size + COL];
        }
      }

      for (int i = 0; i < Tiled_size; ++i) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        c0 = c0 + __shfl_sync(Tiled_mask, a0, i, Tiled_size) * b0[i];
#else
        c0 = c0 + __shfl(a0, i, Tiled_size) * b0[i];
#endif
      }
    }
  }

  __syncthreads();

  if (COL < frame_size) {
    T xt_0 = gate_value[COL + 2 * frame_size];
    T gta_0 = gate_value[COL];
    T htp_0 = 0;
    if (prev_out_value) htp_0 = prev_out_value[COL];
    c0 += xt_0;
    c0 = forward::activation(c0, act_node);
    gate_value[COL + 2 * frame_size] = c0;
    if (origin_mode) {
      output_value[COL] = htp_0 * gta_0 + (1 - gta_0) * c0;
    } else {
      output_value[COL] = c0 * gta_0 + (1 - gta_0) * htp_0;
    }
  }
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class OpStateGrad, bool is_batch, typename T>
__global__ void KeGruBackwardStateGrad(OpStateGrad op_state_grad,
                                       T *gate_value,
                                       T *gate_grad,
                                       const T *prev_out_value,
                                       T *prev_out_grad,
                                       T *output_grad,
                                       int frame_size,
                                       int batch_size,
                                       ActivationType active_node,
                                       bool origin_mode) {
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

  op_state_grad(&r_update_gate_value,
                &r_update_gate_grad,
                &r_frame_state_value,
                &r_frame_state_grad,
                &r_prev_out_value,
                &r_prev_out_grad,
                &r_out_grad,
                active_node,
                origin_mode);

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
__global__ void KeGruBackwardResetGrad(OpResetGrad op_reset_grad,
                                       T *gate_value,
                                       T *gate_grad,
                                       const T *prev_out_value,
                                       T *prev_out_grad,
                                       T *reset_output_grad,
                                       int frame_size,
                                       int batch_size,
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

  op_reset_grad(&r_update_gate_value,
                &r_update_gate_grad,
                &r_reset_gate_value,
                &r_reset_gate_grad,
                &r_prev_out_value,
                &r_prev_out_grad,
                &r_reset_output_grad,
                active_gate);

  gate_grad[frame_idx + frame_size * 0] = r_update_gate_grad;
  gate_grad[frame_idx + frame_size * 1] = r_reset_gate_grad;
  if (prev_out_grad) {
    prev_out_grad[frame_idx] = r_prev_out_grad;
  }
}
}  // namespace detail
}  // namespace funcs
}  // namespace phi

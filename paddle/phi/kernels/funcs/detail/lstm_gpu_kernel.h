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
#include "paddle/phi/kernels/funcs/lstm_compute.h"

namespace phi {
namespace funcs {
namespace detail {

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class T, class Op, bool is_batch>
__global__ void KeLstmForward(Op op,
                              phi::funcs::LstmMetaValue<T> value,
                              int frame_size,
                              int batch_size,
                              T cell_clip,
                              ActivationType active_node,
                              ActivationType active_gate,
                              ActivationType active_state) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;

  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    value.gate_value += batch_idx * frame_size * 4;
    value.output_value += batch_idx * frame_size;
    value.state_value += batch_idx * frame_size;
    value.state_active_value += batch_idx * frame_size;
  }

  T r_state;
  T r_prev_state = 0;
  T r_state_atv;
  T r_out;
  T r_value_in;
  T r_value_ig;
  T r_value_fg;
  T r_value_og;

  T r_checkI = value.check_ig ? value.check_ig[frame_idx] : 0;
  T r_checkF = value.check_fg ? value.check_fg[frame_idx] : 0;
  T r_checkO = value.check_og ? value.check_og[frame_idx] : 0;

  r_value_in = value.gate_value[frame_idx];
  r_value_ig = value.gate_value[frame_idx + frame_size];
  r_value_fg = value.gate_value[frame_idx + frame_size * 2];
  r_value_og = value.gate_value[frame_idx + frame_size * 3];

  if (value.prev_state_value) {
    if (is_batch) value.prev_state_value += batch_idx * frame_size;
    r_prev_state = value.prev_state_value[frame_idx];
  }

  op(&r_value_in,
     &r_value_ig,
     &r_value_fg,
     &r_value_og,
     &r_prev_state,
     &r_state,
     &r_state_atv,
     &r_out,
     &r_checkI,
     &r_checkF,
     &r_checkO,
     &cell_clip,
     active_node,
     active_gate,
     active_state);

  value.gate_value[frame_idx] = r_value_in;
  value.gate_value[frame_idx + frame_size] = r_value_ig;
  value.gate_value[frame_idx + frame_size * 2] = r_value_fg;
  value.gate_value[frame_idx + frame_size * 3] = r_value_og;

  value.state_value[frame_idx] = r_state;
  value.state_active_value[frame_idx] = r_state_atv;
  value.output_value[frame_idx] = r_out;
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <class T, class Op, bool is_batch>
__global__ void KeLstmBackward(Op op,
                               phi::funcs::LstmMetaValue<T> value,
                               phi::funcs::LstmMetaGrad<T> grad,
                               int frame_size,
                               int batch_size,
                               T cell_clip,
                               ActivationType active_node,
                               ActivationType active_gate,
                               ActivationType active_state) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;

  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    value.gate_value += batch_idx * frame_size * 4;
    value.state_value += batch_idx * frame_size;
    value.state_active_value += batch_idx * frame_size;
    grad.gate_grad += batch_idx * frame_size * 4;
    grad.state_grad += batch_idx * frame_size;
    grad.output_grad += batch_idx * frame_size;
  }

  T r_value_in;
  T r_value_ig;
  T r_value_fg;
  T r_value_og;
  T r_grad_in;
  T r_grad_ig;
  T r_grad_fg;
  T r_grad_og;
  T r_prev_state = 0;
  T r_prev_state_grad;
  T r_state;
  T r_state_grad;
  T r_state_atv;
  T r_output_grad;
  T r_checkI = value.check_ig ? value.check_ig[frame_idx] : 0;
  T r_checkF = value.check_fg ? value.check_fg[frame_idx] : 0;
  T r_checkO = value.check_og ? value.check_og[frame_idx] : 0;

  T r_checkIGrad;
  T r_checkFGrad;
  T r_checkOGrad;

  r_value_in = value.gate_value[frame_idx];
  r_value_ig = value.gate_value[frame_idx + frame_size];
  r_value_fg = value.gate_value[frame_idx + frame_size * 2];
  r_value_og = value.gate_value[frame_idx + frame_size * 3];
  r_state = value.state_value[frame_idx];
  r_state_atv = value.state_active_value[frame_idx];
  r_output_grad = grad.output_grad[frame_idx];
  r_state_grad = grad.state_grad[frame_idx];

  if (value.prev_state_value) {
    if (is_batch) value.prev_state_value += batch_idx * frame_size;
    r_prev_state = value.prev_state_value[frame_idx];
  }

  op(&r_value_in,
     &r_value_ig,
     &r_value_fg,
     &r_value_og,
     &r_grad_in,
     &r_grad_ig,
     &r_grad_fg,
     &r_grad_og,
     &r_prev_state,
     &r_prev_state_grad,
     &r_state,
     &r_state_grad,
     &r_state_atv,
     &r_output_grad,
     &r_checkI,
     &r_checkF,
     &r_checkO,
     &r_checkIGrad,
     &r_checkFGrad,
     &r_checkOGrad,
     &cell_clip,
     active_node,
     active_gate,
     active_state);

  grad.gate_grad[frame_idx] = r_grad_in;
  grad.gate_grad[frame_idx + frame_size] = r_grad_ig;
  grad.gate_grad[frame_idx + frame_size * 2] = r_grad_fg;
  grad.gate_grad[frame_idx + frame_size * 3] = r_grad_og;
  grad.state_grad[frame_idx] = r_state_grad;
  if (grad.prev_state_grad) {
    if (is_batch) grad.prev_state_grad += batch_idx * frame_size;
    grad.prev_state_grad[frame_idx] = r_prev_state_grad;
  }

  if (is_batch) {
    if (value.prev_state_value) {
      if (grad.check_ig_grad)
        phi::CudaAtomicAdd(grad.check_ig_grad + frame_idx, r_checkIGrad);
      if (grad.check_fg_grad)
        phi::CudaAtomicAdd(grad.check_fg_grad + frame_idx, r_checkFGrad);
    }
    if (grad.check_og_grad)
      phi::CudaAtomicAdd(grad.check_og_grad + frame_idx, r_checkOGrad);
  } else {
    if (value.prev_state_value) {
      if (grad.check_ig_grad) grad.check_ig_grad[frame_idx] += r_checkIGrad;
      if (grad.check_fg_grad) grad.check_fg_grad[frame_idx] += r_checkFGrad;
    }
    if (grad.check_og_grad) grad.check_og_grad[frame_idx] += r_checkOGrad;
  }
}

template <class T, class Op>
void gpu_lstm_forward(const phi::DeviceContext& context,
                      Op op,
                      phi::funcs::LstmMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      ActivationType active_node,
                      ActivationType active_gate,
                      ActivationType active_state) {
  dim3 threads;
  dim3 grid;
  if (batch_size == 1) {
    int frame_per_block = frame_size <= 1024 ? frame_size : 1024;
    int frame_blocks = (frame_size + 1024 - 1) / 1024;
    threads = dim3(frame_per_block, 1);
    grid = dim3(frame_blocks, 1);
  } else {
    /* frame_per_block = 32 batch_per_block = 16 */
    threads = dim3(32, 16);
    grid = dim3((frame_size + 32 - 1) / 32, (batch_size + 16 - 1) / 16);
  }

  auto stream = reinterpret_cast<const phi::GPUContext&>(context).stream();
  if (batch_size == 1) {
    KeLstmForward<T,
                  Op,
                  /* is_batch= */ false>
        <<<grid, threads, 0, stream>>>(op,
                                       value,
                                       frame_size,
                                       batch_size,
                                       cell_clip,
                                       active_node,
                                       active_gate,
                                       active_state);
  } else {
    KeLstmForward<T,
                  Op,
                  /* is_batch= */ true>
        <<<grid, threads, 0, stream>>>(op,
                                       value,
                                       frame_size,
                                       batch_size,
                                       cell_clip,
                                       active_node,
                                       active_gate,
                                       active_state);
  }
}

template <class T, class Op>
void gpu_lstm_backward(const phi::DeviceContext& context,
                       Op op,
                       phi::funcs::LstmMetaValue<T> value,
                       phi::funcs::LstmMetaGrad<T> grad,
                       int frame_size,
                       int batch_size,
                       T cell_clip,
                       ActivationType active_node,
                       ActivationType active_gate,
                       ActivationType active_state) {
  dim3 threads;
  dim3 grid;
  if (batch_size == 1) {
    int frame_per_block = frame_size <= 1024 ? frame_size : 1024;
    int frame_blocks = (frame_size + 1024 - 1) / 1024;
    threads = dim3(frame_per_block, 1);
    grid = dim3(frame_blocks, 1);
  } else {
    /* frame_per_block = 32 batch_per_block = 16 */
    threads = dim3(32, 16);
    grid = dim3((frame_size + 32 - 1) / 32, (batch_size + 16 - 1) / 16);
  }

  auto stream = reinterpret_cast<const phi::GPUContext&>(context).stream();
  if (batch_size == 1) {
    KeLstmBackward<T,
                   Op,
                   /* is_batch= */ false>
        <<<grid, threads, 0, stream>>>(op,
                                       value,
                                       grad,
                                       frame_size,
                                       batch_size,
                                       cell_clip,
                                       active_node,
                                       active_gate,
                                       active_state);
  } else {
    KeLstmBackward<T,
                   Op,
                   /* is_batch= */ true>
        <<<grid, threads, 0, stream>>>(op,
                                       value,
                                       grad,
                                       frame_size,
                                       batch_size,
                                       cell_clip,
                                       active_node,
                                       active_gate,
                                       active_state);
  }
}

}  // namespace detail
}  // namespace funcs
}  // namespace phi

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
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/gru_compute.h"

namespace paddle {
namespace operators {
namespace math {
namespace detail {

#ifndef __NVCC__

template <class OpResetOutput, typename T>
void hl_naive_gru_forward_reset_output(OpResetOutput op_reset_output,
                                       T *gate_value, T *reset_output_value,
                                       T *prev_output_value, int frame_size,
                                       ActivationType active_gate) {
  T r_value_update_gate;
  T r_value_reset_gate;
  T r_value_reset_output;
  T r_prev_out = 0;
  T *update_gate = gate_value;
  T *reset_gate = gate_value + frame_size;

  for (int i = 0; i < frame_size; i++) {
    r_value_update_gate = update_gate[i];
    r_value_reset_gate = reset_gate[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }

    op_reset_output(&r_value_update_gate, &r_value_reset_gate, &r_prev_out,
                    &r_value_reset_output, active_gate);

    update_gate[i] = r_value_update_gate;
    reset_gate[i] = r_value_reset_gate;
    reset_output_value[i] = r_value_reset_output;
  }
}

template <class OpFinalOutput, typename T>
void hl_naive_gru_forward_final_output(OpFinalOutput op_final_output,
                                       T *gate_value, T *prev_output_value,
                                       T *output_value, int frame_size,
                                       ActivationType active_node,
                                       bool origin_mode) {
  T r_value_update_gate;
  T r_value_frame_state;
  T r_prev_out = 0;
  T r_output;
  T *update_gate = gate_value;
  T *frame_state = gate_value + frame_size * 2;

  for (int i = 0; i < frame_size; i++) {
    r_value_update_gate = update_gate[i];
    r_value_frame_state = frame_state[i];
    if (prev_output_value) {
      r_prev_out = prev_output_value[i];
    }

    op_final_output(&r_value_update_gate, &r_value_frame_state, &r_prev_out,
                    &r_output, active_node, origin_mode);

    frame_state[i] = r_value_frame_state;
    output_value[i] = r_output;
  }
}

template <class OpResetOutput, typename T>
void hl_avx_gru_forward_reset_output(OpResetOutput op_reset_output,
                                     T *gate_value, T *reset_output_value,
                                     T *prev_output_value, int frame_size,
                                     ActivationType active_gate) {
#ifdef __AVX__
  __m256 r_value_update_gate, r_value_update_gate_last = _mm256_set1_ps(0.0f);
  __m256 r_value_reset_gate, r_value_reset_gate_last = _mm256_set1_ps(0.0f);
  __m256 r_value_reset_output;
  __m256 r_prev_out = _mm256_set1_ps(0.0f),
         r_prev_out_last = _mm256_set1_ps(0.0f);
  T *update_gate = gate_value;
  T *reset_gate = gate_value + frame_size;
  int block = 8;
  const int n = frame_size;
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;

  if (rest > 0) {
    i = n - block;
    r_value_update_gate_last =
        _mm256_loadu_ps((const float *)(update_gate + i));
    r_value_reset_gate_last = _mm256_loadu_ps((const float *)(reset_gate + i));
    if (prev_output_value) {
      r_prev_out_last = _mm256_loadu_ps((const float *)(prev_output_value + i));
    }
  }

  for (i = 0; i < end; i += block) {
    r_value_update_gate = _mm256_loadu_ps((const float *)(update_gate + i));
    r_value_reset_gate = _mm256_loadu_ps((const float *)(reset_gate + i));
    if (prev_output_value) {
      r_prev_out = _mm256_loadu_ps((const float *)(prev_output_value + i));
    }

    op_reset_output(&r_value_update_gate, &r_value_reset_gate, &r_prev_out,
                    &r_value_reset_output, active_gate);

    _mm256_storeu_ps(reinterpret_cast<float *>(update_gate + i),
                     r_value_update_gate);
    _mm256_storeu_ps(reinterpret_cast<float *>(reset_gate + i),
                     r_value_reset_gate);
    _mm256_storeu_ps(reinterpret_cast<float *>(reset_output_value + i),
                     r_value_reset_output);
  }

  if (rest > 0) {
    i = n - block;

    op_reset_output(&r_value_update_gate_last, &r_value_reset_gate_last,
                    &r_prev_out_last, &r_value_reset_output, active_gate);

    _mm256_storeu_ps(reinterpret_cast<float *>(update_gate + i),
                     r_value_update_gate_last);
    _mm256_storeu_ps(reinterpret_cast<float *>(reset_gate + i),
                     r_value_reset_gate_last);
    _mm256_storeu_ps(reinterpret_cast<float *>(reset_output_value + i),
                     r_value_reset_output);
  }
#endif
}

template <class OpFinalOutput, typename T>
void hl_avx_gru_forward_final_output(OpFinalOutput op_final_output,
                                     T *gate_value, T *prev_output_value,
                                     T *output_value, int frame_size,
                                     ActivationType active_node,
                                     bool origin_mode) {
#ifdef __AVX__
  __m256 r_value_update_gate, r_value_update_gate_last = _mm256_set1_ps(0.0f);
  __m256 r_value_frame_state, r_value_frame_state_last = _mm256_set1_ps(0.0f);
  __m256 r_prev_out = _mm256_set1_ps(0.0f),
         r_prev_out_last = _mm256_set1_ps(0.0f);
  __m256 r_output;
  T *update_gate = gate_value;
  T *frame_state = gate_value + frame_size * 2;
  int block = 8;
  const int n = frame_size;
  const int rest = n % block;
  const int end = n - rest;
  int i = 0;

  if (rest > 0) {
    i = n - block;
    r_value_update_gate_last =
        _mm256_loadu_ps((const float *)(update_gate + i));
    r_value_frame_state_last =
        _mm256_loadu_ps((const float *)(frame_state + i));
    if (prev_output_value) {
      r_prev_out_last = _mm256_loadu_ps((const float *)(prev_output_value + i));
    }
  }

  for (i = 0; i < end; i += block) {
    r_value_update_gate = _mm256_loadu_ps((const float *)(update_gate + i));
    r_value_frame_state = _mm256_loadu_ps((const float *)(frame_state + i));
    if (prev_output_value) {
      r_prev_out = _mm256_loadu_ps((const float *)(prev_output_value + i));
    }

    op_final_output(&r_value_update_gate, &r_value_frame_state, &r_prev_out,
                    &r_output, active_node, origin_mode);

    _mm256_storeu_ps(reinterpret_cast<float *>(frame_state + i),
                     r_value_frame_state);
    _mm256_storeu_ps(reinterpret_cast<float *>(output_value + i), r_output);
  }

  if (rest > 0) {
    i = n - block;
    op_final_output(&r_value_update_gate_last, &r_value_frame_state_last,
                    &r_prev_out_last, &r_output, active_node, origin_mode);

    _mm256_storeu_ps(reinterpret_cast<float *>(frame_state + i),
                     r_value_frame_state_last);
    _mm256_storeu_ps(reinterpret_cast<float *>(output_value + i), r_output);
  }

#endif
}

template <class OpResetOutput, typename T>
inline void forward_reset_output(OpResetOutput op_reset_output,
                                 GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_gate) {
  for (int b = 0; b < batch_size; b++) {
    if (OpResetOutput::avx && (frame_size > static_cast<int>(8 - 1)) &&
        (sizeof(T) == 4)) {
      hl_avx_gru_forward_reset_output(
          op_reset_output, value.gate_value, value.reset_output_value,
          value.prev_out_value, frame_size, active_gate);
    } else {
      hl_naive_gru_forward_reset_output(
          op_reset_output, value.gate_value, value.reset_output_value,
          value.prev_out_value, frame_size, active_gate);
    }

    value.gate_value += frame_size * 3;
    value.reset_output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

template <class OpFinalOutput, typename T>
inline void forward_final_output(OpFinalOutput op_final_output,
                                 GRUMetaValue<T> value, int frame_size,
                                 int batch_size, ActivationType active_node,
                                 bool origin_mode) {
  for (int b = 0; b < batch_size; b++) {
    if (OpFinalOutput::avx && (frame_size > static_cast<int>(8 - 1)) &&
        (sizeof(T) == 4)) {
      hl_avx_gru_forward_final_output(op_final_output, value.gate_value,
                                      value.prev_out_value, value.output_value,
                                      frame_size, active_node, origin_mode);
    } else {
      hl_naive_gru_forward_final_output(
          op_final_output, value.gate_value, value.prev_out_value,
          value.output_value, frame_size, active_node, origin_mode);
    }

    value.gate_value += frame_size * 3;
    value.output_value += frame_size;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }
  }
}

template <class OpStateGrad, typename T>
void hl_naive_gru_backward_state_grad(OpStateGrad op_state_grad, T *gate_value,
                                      T *gate_grad, T *prev_out_value,
                                      T *prev_out_grad, T *output_grad,
                                      int frame_size,
                                      ActivationType active_node,
                                      bool origin_mode) {
  T r_update_gate_value;
  T r_update_gate_grad;
  T r_frame_state_value;
  T r_frame_state_grad;
  T r_out_grad;
  T r_prev_out_value = 0;
  T r_prev_out_grad = 0;
  T *update_gate_value = gate_value;
  T *update_gate_grad = gate_grad;
  T *frame_state_value = gate_value + frame_size * 2;
  T *frame_state_grad = gate_grad + frame_size * 2;

  for (int i = 0; i < frame_size; i++) {
    r_update_gate_value = update_gate_value[i];
    r_frame_state_value = frame_state_value[i];
    r_out_grad = output_grad[i];
    if (prev_out_value) {
      r_prev_out_value = prev_out_value[i];
    }
    if (prev_out_grad) {
      r_prev_out_grad = prev_out_grad[i];
    }

    op_state_grad(&r_update_gate_value, &r_update_gate_grad,
                  &r_frame_state_value, &r_frame_state_grad, &r_prev_out_value,
                  &r_prev_out_grad, &r_out_grad, active_node, origin_mode);

    update_gate_grad[i] = r_update_gate_grad;
    frame_state_grad[i] = r_frame_state_grad;
    if (prev_out_grad) {
      prev_out_grad[i] = r_prev_out_grad;
    }
  }
}

template <class OpResetGrad, typename T>
void hl_naive_gru_backward_reset_grad(OpResetGrad op_reset_grad, T *gate_value,
                                      T *gate_grad, T *prev_out_value,
                                      T *prev_out_grad, T *reset_output_grad,
                                      int frame_size,
                                      ActivationType active_gate) {
  T r_update_gate_value;
  T r_update_gate_grad;
  T r_reset_gate_value;
  T r_reset_gate_grad;
  T r_reset_output_grad = 0;
  T r_prev_out_value = 0;
  T r_prev_out_grad = 0;
  T *update_gate_value = gate_value;
  T *update_gate_grad = gate_grad;
  T *reset_gate_value = gate_value + frame_size;
  T *reset_gate_grad = gate_grad + frame_size;

  for (int i = 0; i < frame_size; i++) {
    r_update_gate_value = update_gate_value[i];
    r_update_gate_grad = update_gate_grad[i];
    r_reset_gate_value = reset_gate_value[i];

    if (prev_out_value && prev_out_grad) {
      r_reset_output_grad = reset_output_grad[i];
    }
    if (prev_out_value) {
      r_prev_out_value = prev_out_value[i];
    }
    if (prev_out_grad) {
      r_prev_out_grad = prev_out_grad[i];
    }

    op_reset_grad(&r_update_gate_value, &r_update_gate_grad,
                  &r_reset_gate_value, &r_reset_gate_grad, &r_prev_out_value,
                  &r_prev_out_grad, &r_reset_output_grad, active_gate);

    update_gate_grad[i] = r_update_gate_grad;
    reset_gate_grad[i] = r_reset_gate_grad;
    if (prev_out_grad) {
      prev_out_grad[i] = r_prev_out_grad;
    }
  }
}

template <class OpStateGrad, typename T>
void hl_avx_gru_backward_state_grad(OpStateGrad op_state_grad, T *gate_value,
                                    T *gate_grad, T *prev_out_value,
                                    T *prev_out_grad, T *output_grad,
                                    int frame_size, ActivationType active_node,
                                    bool origin_mode) {
#ifdef __AVX__
  __m256 r_update_gate_value;
  __m256 r_update_gate_grad;
  __m256 r_frame_state_value;
  __m256 r_frame_state_grad;
  __m256 r_out_grad;
  __m256 r_prev_out_value = _mm256_set1_ps(0.0f);
  __m256 r_prev_out_grad = _mm256_set1_ps(0.0f);
  __m256 *update_gate_value = reinterpret_cast<__m256 *>(gate_value);
  __m256 *update_gate_grad = reinterpret_cast<__m256 *>(gate_grad);
  __m256 *frame_state_value =
      reinterpret_cast<__m256 *>(gate_value + frame_size * 2);
  __m256 *frame_state_grad =
      reinterpret_cast<__m256 *>(gate_grad + frame_size * 2);

  for (int i = 0; i < frame_size / 8; i++) {
    r_update_gate_value = update_gate_value[i];
    r_frame_state_value = frame_state_value[i];
    r_out_grad = (reinterpret_cast<__m256 *>(output_grad))[i];
    if (prev_out_value) {
      r_prev_out_value = (reinterpret_cast<__m256 *>(prev_out_value))[i];
    }
    if (prev_out_grad) {
      r_prev_out_grad = (reinterpret_cast<__m256 *>(prev_out_grad))[i];
    }

    op_state_grad(&r_update_gate_value, &r_update_gate_grad,
                  &r_frame_state_value, &r_frame_state_grad, &r_prev_out_value,
                  &r_prev_out_grad, &r_out_grad, active_node, origin_mode);

    update_gate_grad[i] = r_update_gate_grad;
    frame_state_grad[i] = r_frame_state_grad;
    if (prev_out_grad) {
      (reinterpret_cast<__m256 *>(prev_out_grad))[i] = r_prev_out_grad;
    }
  }
#endif
}

template <class OpResetGrad, typename T>
void hl_avx_gru_backward_reset_grad(OpResetGrad op_reset_grad, T *gate_value,
                                    T *gate_grad, T *prev_out_value,
                                    T *prev_out_grad, T *reset_output_grad,
                                    int frame_size,
                                    ActivationType active_gate) {
#ifdef __AVX__
  __m256 r_update_gate_value;
  __m256 r_update_gate_grad;
  __m256 r_reset_gate_value;
  __m256 r_reset_gate_grad;
  __m256 r_reset_output_grad = _mm256_set1_ps(0.0f);
  __m256 r_prev_out_value = _mm256_set1_ps(0.0f);
  __m256 r_prev_out_grad = _mm256_set1_ps(0.0f);
  __m256 *update_gate_value = reinterpret_cast<__m256 *>(gate_value);
  __m256 *update_gate_grad = reinterpret_cast<__m256 *>(gate_grad);
  __m256 *reset_gate_value =
      reinterpret_cast<__m256 *>(gate_value + frame_size);
  __m256 *reset_gate_grad = reinterpret_cast<__m256 *>(gate_grad + frame_size);

  for (int i = 0; i < frame_size / 8; i++) {
    r_update_gate_value = update_gate_value[i];
    r_update_gate_grad = update_gate_grad[i];
    r_reset_gate_value = reset_gate_value[i];

    if (prev_out_value && prev_out_grad) {
      r_reset_output_grad = (reinterpret_cast<__m256 *>(reset_output_grad))[i];
    }
    if (prev_out_value) {
      r_prev_out_value = (reinterpret_cast<__m256 *>(prev_out_value))[i];
    }
    if (prev_out_grad) {
      r_prev_out_grad = (reinterpret_cast<__m256 *>(prev_out_grad))[i];
    }

    op_reset_grad(&r_update_gate_value, &r_update_gate_grad,
                  &r_reset_gate_value, &r_reset_gate_grad, &r_prev_out_value,
                  &r_prev_out_grad, &r_reset_output_grad, active_gate);

    update_gate_grad[i] = r_update_gate_grad;
    reset_gate_grad[i] = r_reset_gate_grad;
    if (prev_out_grad) {
      (reinterpret_cast<__m256 *>(prev_out_grad))[i] = r_prev_out_grad;
    }
  }
#endif
}

template <class OpStateGrad, typename T>
inline void backward_state_grad(OpStateGrad op_state_grad,
                                GRUMetaValue<T> value, GRUMetaGrad<T> grad,
                                int frame_size, int batch_size,
                                ActivationType active_node, bool origin_mode) {
  for (int b = 0; b < batch_size; b++) {
    if (OpStateGrad::avx && !(frame_size & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_backward_state_grad(op_state_grad, value.gate_value,
                                     grad.gate_grad, value.prev_out_value,
                                     grad.prev_out_grad, grad.output_grad,
                                     frame_size, active_node, origin_mode);
    } else {
      hl_naive_gru_backward_state_grad(op_state_grad, value.gate_value,
                                       grad.gate_grad, value.prev_out_value,
                                       grad.prev_out_grad, grad.output_grad,
                                       frame_size, active_node, origin_mode);
    }

    value.gate_value += frame_size * 3;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }

    grad.gate_grad += frame_size * 3;
    grad.output_grad += frame_size;
    if (grad.prev_out_grad) {
      grad.prev_out_grad += frame_size;
    }
  }
}

template <class OpResetGrad, typename T>
inline void backward_reset_grad(OpResetGrad op_reset_grad,
                                GRUMetaValue<T> value, GRUMetaGrad<T> grad,
                                int frame_size, int batch_size,
                                ActivationType active_gate) {
  for (int b = 0; b < batch_size; b++) {
    if (OpResetGrad::avx && !(frame_size & (8 - 1)) && (sizeof(T) == 4)) {
      hl_avx_gru_backward_reset_grad(
          op_reset_grad, value.gate_value, grad.gate_grad, value.prev_out_value,
          grad.prev_out_grad, grad.reset_output_grad, frame_size, active_gate);
    } else {
      hl_naive_gru_backward_reset_grad(
          op_reset_grad, value.gate_value, grad.gate_grad, value.prev_out_value,
          grad.prev_out_grad, grad.reset_output_grad, frame_size, active_gate);
    }

    value.gate_value += frame_size * 3;
    if (value.prev_out_value) {
      value.prev_out_value += frame_size;
    }

    grad.gate_grad += frame_size * 3;
    grad.reset_output_grad += frame_size;
    if (grad.prev_out_grad) {
      grad.prev_out_grad += frame_size;
    }
  }
}

#endif

}  // namespace detail
}  // namespace math
}  // namespace operators
}  // namespace paddle

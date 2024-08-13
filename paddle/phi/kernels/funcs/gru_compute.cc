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

#include "paddle/phi/kernels/funcs/gru_compute.h"

#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/detail/gru_cpu_kernel.h"
#include "paddle/phi/kernels/funcs/detail/gru_kernel.h"

namespace phi::funcs {

template <typename T>
struct GRUUnitFunctor<phi::CPUContext, T> {
  static void compute(const phi::CPUContext &context,
                      GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate,
                      bool origin_mode) {
#if !defined(__NVCC__) && !defined(__HIPCC___)
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(context);
    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size * 2,
                frame_size,
                1,
                value.prev_out_value,
                frame_size,
                value.gate_weight,
                frame_size * 2,
                1,
                value.gate_value,
                frame_size * 3);
    }

    detail::forward_reset_output<phi::CPUContext>(
        phi::funcs::detail::forward::gru_resetOutput<T>(),
        value,
        frame_size,
        batch_size,
        active_gate,
        true,
        nullptr);

    if (value.prev_out_value) {
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                value.reset_output_value,
                frame_size,
                value.state_weight,
                frame_size,
                1,
                value.gate_value + frame_size * 2,
                frame_size * 3);
    }

    detail::forward_final_output<phi::CPUContext>(
        phi::funcs::detail::forward::gru_finalOutput<T>(),
        value,
        frame_size,
        batch_size,
        active_node,
        origin_mode,
        true,
        nullptr);
#endif
  }
};

template <typename T>
struct GRUUnitGradFunctor<phi::CPUContext, T> {
  static void compute(const phi::CPUContext &context,
                      GRUMetaValue<T> value,
                      GRUMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate,
                      bool origin_mode) {
#if !defined(__NVCC__) && !defined(__HIPCC___)
    detail::backward_state_grad(
        phi::funcs::detail::backward::gru_stateGrad<T>(),
        value,
        grad,
        frame_size,
        batch_size,
        active_node,
        origin_mode);
    auto blas = phi::funcs::GetBlas<phi::CPUContext, T>(context);
    if (value.prev_out_value && grad.prev_out_grad) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.gate_grad + frame_size * 2,
                frame_size * 3,
                value.state_weight,
                frame_size,
                0,
                grad.reset_output_grad,
                frame_size);

      if (grad.state_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  value.reset_output_value,
                  frame_size,
                  grad.gate_grad + frame_size * 2,
                  frame_size * 3,
                  1,
                  grad.state_weight_grad,
                  frame_size);
      }
    }

    detail::backward_reset_grad(
        phi::funcs::detail::backward::gru_resetGrad<T>(),
        value,
        grad,
        frame_size,
        batch_size,
        active_gate);
    if (grad.prev_out_grad && value.prev_out_value) {
      blas.GEMM(false,
                true,
                batch_size,
                frame_size,
                frame_size * 2,
                1,
                grad.gate_grad,
                frame_size * 3,
                value.gate_weight,
                frame_size * 2,
                1,
                grad.prev_out_grad,
                frame_size);

      if (grad.gate_weight_grad) {
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size * 2,
                  batch_size,
                  1,
                  value.prev_out_value,
                  frame_size,
                  grad.gate_grad,
                  frame_size * 3,
                  1,
                  grad.gate_weight_grad,
                  frame_size * 2);
      }
    }
#endif
  }
};

template <typename T>
struct GRUUnitFunctorV2<CPUContext, T> {
  static void compute(const CPUContext &context,
                      GRUMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate) {
#if !defined(__NVCC__) && !defined(__HIPCC___)
    auto blas = phi::funcs::GetBlas<CPUContext, T>(context);
    if (value.prev_out_value) {
      blas.GEMM(CblasNoTrans,
                CblasTrans,
                batch_size,
                frame_size,
                frame_size,
                1,
                value.prev_out_value,
                value.state_weight,
                0,
                value.reset_output_value);
    }
    detail::forward_reset_output(
        phi::funcs::detail::forward::gru_resetOutput<T>(),
        value,
        frame_size,
        batch_size,
        active_gate,
        false,
        &context);

    T *cell_state_value = value.gate_value + 2 * frame_size;
    T *reset_output_value = value.reset_output_value;
    for (int b = 0; b < batch_size; ++b) {
      blas.VADD(
          frame_size, cell_state_value, reset_output_value, cell_state_value);
      cell_state_value += frame_size * 3;
      reset_output_value += frame_size;
    }

    detail::forward_final_output(
        phi::funcs::detail::forward::gru_finalOutput<T>(),
        value,
        frame_size,
        batch_size,
        active_node,
        true,
        false,
        &context);
#endif
  }
};

template <typename T>
struct GRUUnitGradFunctorV2<CPUContext, T> {
  static void compute(const CPUContext &context,
                      GRUMetaValue<T> value,
                      GRUMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      const phi::funcs::detail::ActivationType active_node,
                      const phi::funcs::detail::ActivationType active_gate) {
#if !defined(__NVCC__) && !defined(__HIPCC___)
    // calculate grad_update_gate, grad_frame_state,
    // grad_reset_output, grad_reset_gate
    detail::cpu_gru_backward(context,
                             phi::funcs::detail::backward::gru<T>(),
                             value,
                             grad,
                             frame_size,
                             batch_size,
                             active_node,
                             active_gate);
    auto blas = phi::funcs::GetBlas<CPUContext, T>(context);
    if (grad.prev_out_grad && value.prev_out_value) {
      // update prev_out_grad
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.gate_grad,
                frame_size * 3,
                value.gate_weight,
                frame_size,
                1,
                grad.prev_out_grad,
                frame_size);
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.gate_grad + frame_size,
                frame_size * 3,
                value.gate_weight + frame_size * frame_size,
                frame_size,
                1,
                grad.prev_out_grad,
                frame_size);
      blas.GEMM(false,
                false,
                batch_size,
                frame_size,
                frame_size,
                1,
                grad.reset_output_grad,
                frame_size,
                value.state_weight,
                frame_size,
                1,
                grad.prev_out_grad,
                frame_size);
      // update weight_hh_grad
      if (grad.gate_weight_grad) {
        // reset gate
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  grad.gate_grad,
                  frame_size * 3,
                  value.prev_out_value,
                  frame_size,
                  1,
                  grad.gate_weight_grad,
                  frame_size);
        // update gate
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  grad.gate_grad + frame_size,
                  frame_size * 3,
                  value.prev_out_value,
                  frame_size,
                  1,
                  grad.gate_weight_grad + frame_size * frame_size,
                  frame_size);
        // cell state
        blas.GEMM(true,
                  false,
                  frame_size,
                  frame_size,
                  batch_size,
                  1,
                  grad.reset_output_grad,
                  frame_size,
                  value.prev_out_value,
                  frame_size,
                  1,
                  grad.state_weight_grad,
                  frame_size);
      }
    }
    // update bias_hh_grad
    T *gate_grad = grad.gate_grad;
    T *bias_hh_grad = grad.bias_hh_grad;
    T *state_bias_grad = grad.bias_hh_grad + 2 * frame_size;
    T *reset_output_grad = grad.reset_output_grad;
    for (int b = 0; b < batch_size; ++b) {
      blas.VADD(2 * frame_size, bias_hh_grad, gate_grad, bias_hh_grad);
      blas.VADD(
          frame_size, state_bias_grad, reset_output_grad, state_bias_grad);
      gate_grad += 3 * frame_size;
      reset_output_grad += frame_size;
    }
#endif
  }
};

template struct GRUUnitFunctor<phi::CPUContext, float>;
template struct GRUUnitFunctor<phi::CPUContext, double>;
template struct GRUUnitGradFunctor<phi::CPUContext, float>;
template struct GRUUnitGradFunctor<phi::CPUContext, double>;

template struct GRUUnitFunctorV2<CPUContext, float>;
template struct GRUUnitFunctorV2<CPUContext, double>;
template struct GRUUnitGradFunctorV2<CPUContext, float>;
template struct GRUUnitGradFunctorV2<CPUContext, double>;

}  // namespace phi::funcs

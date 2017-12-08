/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/operators/math/gru_compute.h"
#include "paddle/operators/math/detail/gru_cpu_kernel.h"
#include "paddle/operators/math/detail/gru_kernel.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct GRUUnitFunctor<platform::CPUPlace, T> {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, int frame_size, int batch_size,
                      activation_mode_t active_node,
                      activation_mode_t active_gate) {
#ifndef __NVCC__
    if (value.prev_out_value) {
      math::gemm<platform::CPUPlace, T>(
          context, false, false, batch_size, frame_size * 2, frame_size, 1,
          value.prev_out_value, frame_size, value.gate_weight, frame_size * 2,
          1, value.gate_value, frame_size * 3);
    }

    detail::forward_reset_output(detail::forward::gru_resetOutput<T>(), value,
                                 frame_size, batch_size, active_gate);

    if (value.prev_out_value) {
      math::gemm<platform::CPUPlace, T>(
          context, false, false, batch_size, frame_size, frame_size, 1,
          value.reset_output_value, frame_size, value.state_weight, frame_size,
          1, value.gate_value + frame_size * 2, frame_size * 3);
    }

    detail::forward_final_output(detail::forward::gru_finalOutput<T>(), value,
                                 frame_size, batch_size, active_node);
#endif
  }
};

template <typename T>
struct GRUUnitGradFunctor<platform::CPUPlace, T> {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, hl_gru_grad<T> grad,
                      int frame_size, int batch_size,
                      activation_mode_t active_node,
                      activation_mode_t active_gate) {
#ifndef __NVCC__
    detail::backward_state_grad(detail::backward::gru_stateGrad<T>(), value,
                                grad, frame_size, batch_size, active_node);

    if (value.prev_out_value && grad.prev_out_grad) {
      math::gemm<platform::CPUPlace, T>(
          context, false, true, batch_size, frame_size, frame_size, 1,
          grad.gate_grad + frame_size * 2, frame_size * 3, value.state_weight,
          frame_size, 0, grad.reset_output_grad, frame_size);

      if (grad.state_weight_grad) {
        math::gemm<platform::CPUPlace, T>(
            context, true, false, frame_size, frame_size, batch_size, 1,
            value.reset_output_value, frame_size,
            grad.gate_grad + frame_size * 2, frame_size * 3, 1,
            grad.state_weight_grad, frame_size);
      }
    }

    detail::backward_reset_grad(detail::backward::gru_resetGrad<T>(), value,
                                grad, frame_size, batch_size, active_gate);

    if (grad.prev_out_grad && value.prev_out_value) {
      math::gemm<platform::CPUPlace, T>(
          context, false, true, batch_size, frame_size, frame_size * 2, 1,
          grad.gate_grad, frame_size * 3, value.gate_weight, frame_size * 2, 1,
          grad.prev_out_grad, frame_size);

      if (grad.gate_weight_grad) {
        math::gemm<platform::CPUPlace, T>(
            context, true, false, frame_size, frame_size * 2, batch_size, 1,
            value.prev_out_value, frame_size, grad.gate_grad, frame_size * 3, 1,
            grad.gate_weight_grad, frame_size * 2);
      }
    }
#endif
  }
};

template struct GRUUnitFunctor<platform::CPUPlace, float>;
template struct GRUUnitFunctor<platform::CPUPlace, double>;
template struct GRUUnitGradFunctor<platform::CPUPlace, float>;
template struct GRUUnitGradFunctor<platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle

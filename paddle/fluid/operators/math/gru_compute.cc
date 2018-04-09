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

#include "paddle/fluid/operators/math/gru_compute.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/gru_cpu_kernel.h"
#include "paddle/fluid/operators/math/detail/gru_kernel.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct GRUUnitFunctor<platform::CPUDeviceContext, T> {
  static void compute(const platform::CPUDeviceContext &context,
                      GRUMetaValue<T> value, int frame_size, int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate) {
#ifndef __HIPCC__
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    if (value.prev_out_value) {
      blas.GEMM(false, false, batch_size, frame_size * 2, frame_size, 1,
                value.prev_out_value, frame_size, value.gate_weight,
                frame_size * 2, 1, value.gate_value, frame_size * 3);
    }

    detail::forward_reset_output(detail::forward::gru_resetOutput<T>(), value,
                                 frame_size, batch_size, active_gate);

    if (value.prev_out_value) {
      blas.GEMM(false, false, batch_size, frame_size, frame_size, 1,
                value.reset_output_value, frame_size, value.state_weight,
                frame_size, 1, value.gate_value + frame_size * 2,
                frame_size * 3);
    }

    detail::forward_final_output(detail::forward::gru_finalOutput<T>(), value,
                                 frame_size, batch_size, active_node);
#endif
  }
};

template <typename T>
struct GRUUnitGradFunctor<platform::CPUDeviceContext, T> {
  static void compute(const platform::CPUDeviceContext &context,
                      GRUMetaValue<T> value, GRUMetaGrad<T> grad,
                      int frame_size, int batch_size,
                      const detail::ActivationType active_node,
                      const detail::ActivationType active_gate) {
#ifndef __HIPCC__
    detail::backward_state_grad(detail::backward::gru_stateGrad<T>(), value,
                                grad, frame_size, batch_size, active_node);
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    if (value.prev_out_value && grad.prev_out_grad) {
      blas.GEMM(false, true, batch_size, frame_size, frame_size, 1,
                grad.gate_grad + frame_size * 2, frame_size * 3,
                value.state_weight, frame_size, 0, grad.reset_output_grad,
                frame_size);

      if (grad.state_weight_grad) {
        blas.GEMM(true, false, frame_size, frame_size, batch_size, 1,
                  value.reset_output_value, frame_size,
                  grad.gate_grad + frame_size * 2, frame_size * 3, 1,
                  grad.state_weight_grad, frame_size);
      }
    }

    detail::backward_reset_grad(detail::backward::gru_resetGrad<T>(), value,
                                grad, frame_size, batch_size, active_gate);
    if (grad.prev_out_grad && value.prev_out_value) {
      blas.GEMM(false, true, batch_size, frame_size, frame_size * 2, 1,
                grad.gate_grad, frame_size * 3, value.gate_weight,
                frame_size * 2, 1, grad.prev_out_grad, frame_size);

      if (grad.gate_weight_grad) {
        blas.GEMM(true, false, frame_size, frame_size * 2, batch_size, 1,
                  value.prev_out_value, frame_size, grad.gate_grad,
                  frame_size * 3, 1, grad.gate_weight_grad, frame_size * 2);
      }
    }
#endif
  }
};

template struct GRUUnitFunctor<platform::CPUDeviceContext, float>;
template struct GRUUnitFunctor<platform::CPUDeviceContext, double>;
template struct GRUUnitGradFunctor<platform::CPUDeviceContext, float>;
template struct GRUUnitGradFunctor<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle

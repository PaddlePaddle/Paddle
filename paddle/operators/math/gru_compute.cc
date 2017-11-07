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
                      hl_gru_value<T> value, int frameSize, int batchSize,
                      activation_mode_t active_node,
                      activation_mode_t active_gate) {
#ifndef __NVCC__
    if (value.prevOutValue) {
      math::gemm<platform::CPUPlace, T>(
          context, false, false, batchSize, frameSize * 2, frameSize, 1,
          value.prevOutValue, frameSize, value.gateWeight, frameSize * 2, 1,
          value.gateValue, frameSize * 3);
    }

    detail::forward_reset_output(detail::forward::gru_resetOutput<T>(), value,
                                 frameSize, batchSize, active_gate);

    if (value.prevOutValue) {
      math::gemm<platform::CPUPlace, T>(
          context, false, false, batchSize, frameSize, frameSize, 1,
          value.resetOutputValue, frameSize, value.stateWeight, frameSize, 1,
          value.gateValue + frameSize * 2, frameSize * 3);
    }

    detail::forward_final_output(detail::forward::gru_finalOutput<T>(), value,
                                 frameSize, batchSize, active_node);
#endif
  }
};

template <typename T>
struct GRUUnitGradFunctor<platform::CPUPlace, T> {
  static void compute(const platform::DeviceContext &context,
                      hl_gru_value<T> value, hl_gru_grad<T> grad, int frameSize,
                      int batchSize, activation_mode_t active_node,
                      activation_mode_t active_gate) {
#ifndef __NVCC__
    detail::backward_state_grad(detail::backward::gru_stateGrad<T>(), value,
                                grad, frameSize, batchSize, active_node);

    if (value.prevOutValue && grad.prevOutGrad) {
      math::gemm<platform::CPUPlace, T>(
          context, false, true, batchSize, frameSize, frameSize, 1,
          grad.gateGrad + frameSize * 2, frameSize * 3, value.stateWeight,
          frameSize, 0, grad.resetOutputGrad, frameSize);

      if (grad.stateWeightGrad) {
        math::gemm<platform::CPUPlace, T>(
            context, true, false, frameSize, frameSize, batchSize, 1,
            value.resetOutputValue, frameSize, grad.gateGrad + frameSize * 2,
            frameSize * 3, 1, grad.stateWeightGrad, frameSize);
      }
    }

    detail::backward_reset_grad(detail::backward::gru_resetGrad<T>(), value,
                                grad, frameSize, batchSize, active_gate);

    if (grad.prevOutGrad && value.prevOutValue) {
      math::gemm<platform::CPUPlace, T>(
          context, false, true, batchSize, frameSize, frameSize * 2, 1,
          grad.gateGrad, frameSize * 3, value.gateWeight, frameSize * 2, 1,
          grad.prevOutGrad, frameSize);

      if (grad.gateWeightGrad) {
        math::gemm<platform::CPUPlace, T>(
            context, true, false, frameSize, frameSize * 2, batchSize, 1,
            value.prevOutValue, frameSize, grad.gateGrad, frameSize * 3, 1,
            grad.gateWeightGrad, frameSize * 2);
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

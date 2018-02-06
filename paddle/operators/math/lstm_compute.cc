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

#include "paddle/operators/math/lstm_compute.h"
#include "paddle/operators/math/detail/lstm_cpu_kernel.h"
#include "paddle/operators/math/detail/lstm_kernel.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
struct LstmUnitFunctor<platform::CPUPlace, T> {
  static void compute(const platform::DeviceContext& context,
                      LstmMetaValue<T> value, int frame_size, int batch_size,
                      const std::string& gate_act, const std::string& cell_act,
                      const std::string& cand_act) {
    for (int b = 0; b < batch_size; b++) {
      detail::cpu_lstm_forward(detail::forward::lstm<T>(), value, frame_size,
                               ActiveType(cand_act), ActiveType(gate_act),
                               ActiveType(cell_act));
      value.gate_value += frame_size * 4;
      value.state_value += frame_size;
      value.state_active_value += frame_size;
      value.output_value += frame_size;
      if (value.prev_state_value) {
        value.prev_state_value += frame_size;
      }
    }
  }
};

template <class T>
struct LstmUnitGradFunctor<platform::CPUPlace, T> {
  static void compute(const platform::DeviceContext& context,
                      LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                      int frame_size, int batch_size,
                      const std::string& gate_act, const std::string& cell_act,
                      const std::string& cand_act) {
    for (int b = 0; b < batch_size; b++) {
      detail::cpu_lstm_backward(detail::backward::lstm<T>(), value, grad,
                                frame_size, ActiveType(cand_act),
                                ActiveType(gate_act), ActiveType(cell_act));

      value.gate_value += frame_size * 4;
      value.state_value += frame_size;
      value.state_active_value += frame_size;
      value.output_value += frame_size;
      if (value.prev_state_value) {
        value.prev_state_value += frame_size;
      }

      grad.gate_grad += frame_size * 4;
      grad.state_grad += frame_size;
      grad.state_active_grad += frame_size;
      grad.output_grad += frame_size;
      if (grad.prev_state_grad) {
        grad.prev_state_grad += frame_size;
      }
    }
  }
};

template class LstmUnitFunctor<platform::CPUPlace, float>;
template class LstmUnitFunctor<platform::CPUPlace, double>;
template class LstmUnitGradFunctor<platform::CPUPlace, float>;
template class LstmUnitGradFunctor<platform::CPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle

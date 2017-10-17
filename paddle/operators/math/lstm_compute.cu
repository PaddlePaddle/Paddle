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

#include "paddle/operators/math/detail/lstm_gpu_kernel.h"
#include "paddle/operators/math/detail/lstm_kernel.h"
#include "paddle/operators/math/lstm_compute.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
struct LstmUnitFunctor<platform::GPUPlace, T> {
  static void compute(const platform::DeviceContext& context,
                      LstmMetaValue<T> value, int frame_size, int batch_size,
                      std::string gate_act, std::string cell_act,
                      std::string cand_act) {
    for (int b = 0; b < batch_size; b++) {
      detail::gpu_lstm_forward(context, detail::forward::lstm<T>(), value,
                               frame_size, batch_size, ActiveType(cand_act),
                               ActiveType(gate_act), ActiveType(cell_act));
      value.gateValue += frame_size * 4;
      value.stateValue += frame_size;
      value.stateActiveValue += frame_size;
      value.outputValue += frame_size;
      if (value.prevStateValue) {
        value.prevStateValue += frame_size;
      }
    }
  }
};

template <class T>
struct LstmUnitGradFunctor<platform::GPUPlace, T> {
  static void compute(const platform::DeviceContext& context,
                      LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                      int frame_size, int batch_size, std::string gate_act,
                      std::string cell_act, std::string cand_act) {
    for (int b = 0; b < batch_size; b++) {
      detail::gpu_lstm_backward(context, detail::backward::lstm<T>(), value,
                                grad, frame_size, batch_size,
                                ActiveType(cand_act), ActiveType(gate_act),
                                ActiveType(cell_act));

      value.gateValue += frame_size * 4;
      value.stateValue += frame_size;
      value.stateActiveValue += frame_size;
      value.outputValue += frame_size;
      if (value.prevStateValue) {
        value.prevStateValue += frame_size;
      }

      grad.gateGrad += frame_size * 4;
      grad.stateGrad += frame_size;
      grad.stateActiveGrad += frame_size;
      grad.outputGrad += frame_size;
      if (grad.prevStateGrad) {
        grad.prevStateGrad += frame_size;
      }
    }
  }
};

template class LstmUnitFunctor<platform::GPUPlace, float>;
template class LstmUnitGradFunctor<platform::GPUPlace, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle

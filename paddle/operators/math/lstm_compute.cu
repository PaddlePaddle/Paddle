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

#include "LstmCompute.h"
#include "paddle/operators/math/detail/lstm_cpu_kernel.h"
#include "paddle/operators/math/detail/lstm_kernel.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
struct LstmUnitFunctor<platform::GPUPlace, T> {
  static void compute(lstm_value value, int frame_size, int batch_size,
                      std::string gate_act, std::string cell_act,
                      std::string cand_act) {
    for (int b = 0; b < batch_size; b++) {
      detail::gpu_lstm_forward(detail::forward::lstm<T>(), value, frameSize,
                               ActiveType(cand_act), ActiveType(gate_act),
                               ActiveType(cell_act));
      value.gateValue += frameSize * 4;
      value.stateValue += frameSize;
      value.stateActiveValue += frameSize;
      value.outputValue += frameSize;
      if (value.prevStateValue) {
        value.prevStateValue += frameSize;
      }
    }
  }
};

template <class T>
struct LstmUnitGradFunctor<platform::GPUPlace, T> {
  static void compute(lstm_value value, lstm_grad grad, int frame_size,
                      int batch_size, std::string gate_act,
                      std::string cell_act, std::string cand_act) {
    for (int b = 0; b < batchSize; b++) {
      detail::gpu_lstm_backward(detail::backward::lstm<T>(), value, grad,
                                frameSize, ActiveType(cand_act),
                                ActiveType(gate_act), ActiveType(cell_act));

      value.gateValue += frameSize * 4;
      value.stateValue += frameSize;
      value.stateActiveValue += frameSize;
      value.outputValue += frameSize;
      if (value.prevStateValue) {
        value.prevStateValue += frameSize;
      }

      grad.gateGrad += frameSize * 4;
      grad.stateGrad += frameSize;
      grad.stateActiveGrad += frameSize;
      grad.outputGrad += frameSize;
      if (grad.prevStateGrad) {
        grad.prevStateGrad += frameSize;
      }
    }
  };

}  // namespace math
}  // namespace operators
}  // namespace paddle

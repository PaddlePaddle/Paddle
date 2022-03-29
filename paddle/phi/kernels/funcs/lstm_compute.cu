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

#include "paddle/phi/kernels/funcs/detail/lstm_gpu_kernel.h"
#include "paddle/phi/kernels/funcs/detail/lstm_kernel.h"
#include "paddle/phi/kernels/funcs/lstm_compute.h"

namespace phi {
namespace funcs {

template <class T>
struct LstmUnitFunctor<paddle::platform::CUDADeviceContext, T> {
  static void compute(const paddle::platform::CUDADeviceContext& context,
                      LstmMetaValue<T> value,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      const phi::funcs::detail::ActivationType& gate_act,
                      const phi::funcs::detail::ActivationType& cell_act,
                      const phi::funcs::detail::ActivationType& cand_act,
                      bool old_api_version = true) {
    detail::gpu_lstm_forward<T>(context,
                                phi::funcs::detail::forward::lstm<T>(),
                                value,
                                frame_size,
                                batch_size,
                                cell_clip,
                                cand_act,
                                gate_act,
                                cell_act);
  }
};

template <class T>
struct LstmUnitGradFunctor<paddle::platform::CUDADeviceContext, T> {
  static void compute(const paddle::platform::CUDADeviceContext& context,
                      LstmMetaValue<T> value,
                      LstmMetaGrad<T> grad,
                      int frame_size,
                      int batch_size,
                      T cell_clip,
                      const phi::funcs::detail::ActivationType& gate_act,
                      const phi::funcs::detail::ActivationType& cell_act,
                      const phi::funcs::detail::ActivationType& cand_act,
                      bool old_api_version = true) {
    detail::gpu_lstm_backward(context,
                              phi::funcs::detail::backward::lstm<T>(),
                              value,
                              grad,
                              frame_size,
                              batch_size,
                              cell_clip,
                              cand_act,
                              gate_act,
                              cell_act);
  }
};

template class LstmUnitFunctor<paddle::platform::CUDADeviceContext, float>;
template class LstmUnitFunctor<paddle::platform::CUDADeviceContext, double>;
template class LstmUnitGradFunctor<paddle::platform::CUDADeviceContext, float>;
template class LstmUnitGradFunctor<paddle::platform::CUDADeviceContext, double>;

}  // namespace funcs
}  // namespace phi

/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fake_dequantize_functor.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void DequantizeFunctor<Context, T>::operator()(const Context& dev_ctx,
                                               const DenseTensor* in,
                                               const DenseTensor* scale,
                                               T max_range,
                                               DenseTensor* out) {
  auto in_e = phi::EigenVector<T>::Flatten(*in);
  const T* scale_factor = scale->data<T>();
  auto out_e = phi::EigenVector<T>::Flatten(*out);

  auto& dev = *dev_ctx.eigen_device();
  out_e.device(dev) = in_e * scale_factor[0] / max_range;
}

template class DequantizeFunctor<CPUContext, float>;
template class DequantizeFunctor<CPUContext, double>;

}  // namespace funcs
}  // namespace phi

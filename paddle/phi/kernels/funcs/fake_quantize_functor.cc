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

#include "paddle/phi/kernels/funcs/fake_quantize_functor.h"

namespace phi {
namespace funcs {

template <typename Context, typename T>
void FindAbsMaxFunctor<Context, T>::operator()(const Context &ctx,
                                               const T *in,
                                               const int num,
                                               T *out) {
  *out = std::abs(*(std::max_element(in + 0, in + num, Compare<T>())));
}

template <typename Context, typename T>
void ClipAndFakeQuantFunctor<Context, T>::operator()(const Context &ctx,
                                                     const DenseTensor &in,
                                                     const DenseTensor &scale,
                                                     const int bin_cnt,
                                                     const int round_type,
                                                     DenseTensor *out) {
  T s = scale.data<T>()[0];
  T inv_s = inverse(s);
  phi::Transform<Context> trans;
  if (round_type == 0) {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          QuantTensorFunctor<T>(static_cast<T>(bin_cnt), inv_s));
  } else {
    trans(ctx,
          in.data<T>(),
          in.data<T>() + in.numel(),
          ctx.template Alloc<T>(out),
          phi::ClipFunctor<T>(-s, s));
    auto out_e = EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
  }
}

template class FindAbsMaxFunctor<CPUContext, float>;
template class ClipAndFakeQuantFunctor<CPUContext, float>;

}  // namespace funcs
}  // namespace phi

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

template <typename Context, typename T>
void FindMovingAverageAbsMaxFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &in_accum,
    const DenseTensor &in_state,
    const T *cur_scale,
    const float rate,
    DenseTensor *out_state,
    DenseTensor *out_accum,
    DenseTensor *out_scale) {
  T accum = in_accum.data<T>()[0];
  T state = in_state.data<T>()[0];
  T scale = cur_scale[0];

  state = rate * state + 1;
  accum = rate * accum + scale;
  scale = accum / state;

  T *out_state_data = ctx.template Alloc<T>(out_state);
  T *out_accum_data = ctx.template Alloc<T>(out_accum);
  T *out_scale_data = ctx.template Alloc<T>(out_scale);

  out_state_data[0] = state;
  out_accum_data[0] = accum;
  out_scale_data[0] = scale;
}

template <typename Context, typename T>
void FindRangeAbsMaxFunctor<Context, T>::operator()(
    const Context &ctx,
    const DenseTensor &cur_scale,
    const DenseTensor &last_scale,
    const DenseTensor &iter,
    const int window_size,
    DenseTensor *scales_arr,
    DenseTensor *out_scale) {
  T *scale_arr_data = ctx.template Alloc<T>(scales_arr);
  int64_t it = iter.data<int64_t>()[0];
  int idx = static_cast<int>(it % window_size);
  T removed = scale_arr_data[idx];
  T cur = cur_scale.data<T>()[0];
  scale_arr_data[idx] = cur;

  T max = last_scale.data<T>()[0];
  if (max < cur) {
    max = cur;
  } else if (fabs(removed - max) < 1e-6) {
    int size = static_cast<int>((it > window_size) ? window_size : it);
    phi::funcs::FindAbsMaxFunctor<Context, T>()(
        ctx, scale_arr_data, size, &max);
  }
  T *out_scale_data = ctx.template Alloc<T>(out_scale);
  out_scale_data[0] = max;
}

template class FindAbsMaxFunctor<CPUContext, float>;
template class ClipAndFakeQuantFunctor<CPUContext, float>;
template class FindMovingAverageAbsMaxFunctor<CPUContext, float>;
template class FindRangeAbsMaxFunctor<CPUContext, float>;

}  // namespace funcs
}  // namespace phi

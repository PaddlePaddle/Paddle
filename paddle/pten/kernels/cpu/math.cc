//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/pten/kernels/cpu/math.h"

#include "paddle/pten/kernels/functions/cpu/elementwise.h"
#include "paddle/pten/kernels/functions/eigen/mean.h"
#include "paddle/pten/kernels/functions/eigen/scale.h"
#include "paddle/pten/kernels/functions/eigen/sign.h"
#include "paddle/pten/kernels/functions/general/elementwise_functor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"

namespace pten {

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Sign<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  eigen::Mean<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Scale(const CPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  eigen::Scale<CPUContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

// TODO(chenweihang): now the ScaleTensor's dtype are same as x, so we cannot
// register its dtype def
template <typename T>
void ScaleHost(const CPUContext& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& scale,
               float bias,
               bool bias_after_scale,
               DenseTensor* out) {
  eigen::Scale<CPUContext, T>(dev_ctx,
                              x,
                              static_cast<float>(*scale.data<T>()),
                              bias,
                              bias_after_scale,
                              out);
}

template <typename T>
void ElementwiseAdd(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  if (x.dims() == y.dims()) {
    SameDimsElementwiseCompute<general::SameDimsAddFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<general::AddFunctor<T>, T>(
          dev_ctx, x, y, axis, general::AddFunctor<T>(), out);
    } else {
      ElementwiseCompute<general::InverseAddFunctor<T>, T>(
          dev_ctx, x, y, axis, general::InverseAddFunctor<T>(), out);
    }
  }
}

template <typename T>
void ElementwiseSub(const CPUContext& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  if (x.dims() == y.dims()) {
    SameDimsElementwiseCompute<general::SameDimsSubFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      ElementwiseCompute<general::SubFunctor<T>, T>(
          dev_ctx, x, y, axis, general::SubFunctor<T>(), out);
    } else {
      ElementwiseCompute<general::InverseSubFunctor<T>, T>(
          dev_ctx, x, y, axis, general::InverseSubFunctor<T>(), out);
    }
  }
}

}  // namespace pten

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(MathCPU);

using complex64 = ::paddle::platform::complex<float>;
using complex128 = ::paddle::platform::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::paddle::platform::bfloat16;

PT_REGISTER_KERNEL("sign", CPU, ANY, pten::Sign, float, double) {}
PT_REGISTER_KERNEL(
    "mean", CPU, ANY, pten::Mean, float, double, paddle::platform::bfloat16) {}
PT_REGISTER_KERNEL("scale",
                   CPU,
                   ANY,
                   pten::Scale,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.host",
                   CPU,
                   ANY,
                   pten::ScaleHost,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetBackend(pten::Backend::CPU);
}
PT_REGISTER_KERNEL("elementwise_add",
                   CPU,
                   ANY,
                   pten::ElementwiseAdd,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PT_REGISTER_KERNEL("elementwise_sub",
                   CPU,
                   ANY,
                   pten::ElementwiseSub,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}

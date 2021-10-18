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

#include "paddle/tcmpt/kernels/cpu/math.h"

#include "paddle/tcmpt/kernels/common/eigen/mean.h"
#include "paddle/tcmpt/kernels/common/eigen/scale.h"
#include "paddle/tcmpt/kernels/common/eigen/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/bfloat16.h"

namespace pt {

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

}  // namespace pt

// TODO(chenweihang): replace by better impl
PT_REGISTER_MODULE(MathCPU);

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::paddle::platform::bfloat16;

PT_REGISTER_KERNEL("sign", CPU, NCHW, pt::Sign, float, double) {}
PT_REGISTER_KERNEL("mean", CPU, NCHW, pt::Mean, float, double) {}
PT_REGISTER_KERNEL("scale",
                   CPU,
                   NCHW,
                   pt::Scale,
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
                   NCHW,
                   pt::ScaleHost,
                   float,
                   double,
                   paddle::platform::bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1).SetBackend(pt::Backend::kCPU);
}

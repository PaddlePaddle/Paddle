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

#include "paddle/tcmpt/cpu/math.h"

#include "paddle/tcmpt/eigen/scale.h"
#include "paddle/tcmpt/eigen/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"

namespace pt {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = paddle::framework::EigenScalar<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = paddle::framework::EigenVector<T, MajorType, IndexType>;

template <typename T>
void Sign(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  module::Sign<CPUContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUContext& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  out->mutable_data<T>();
  auto x_data = EigenVector<T>::Flatten(x);
  auto y_data = EigenScalar<T>::From(*out);
  auto& place = *dev_ctx.eigen_device();
  y_data.device(place) = x_data.mean();
}

template <typename T>
void Scale(const CPUContext& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  module::Scale<CPUContext, T>(dev_ctx, x, scale, bias, bias_after_scale, out);
}

template <typename T>
void ScaleSelectedRows(const CPUContext& dev_ctx,
                       const SelectedRowsTensor& x,
                       float scale,
                       float bias,
                       bool bias_after_scale,
                       SelectedRowsTensor* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  Scale<T>(
      dev_ctx, x.value(), scale, bias, bias_after_scale, out->mutable_value());
}

template <typename T>
void ScaleDynamicAttr(const CPUContext& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& scale,
                      float bias,
                      bool bias_after_scale,
                      DenseTensor* out) {
  module::Scale<CPUContext, T>(
      dev_ctx, x, *scale.data<float>(), bias, bias_after_scale, out);
}

template <typename T>
void ScaleSelectedRowsDynamicAttr(const CPUContext& dev_ctx,
                                  const SelectedRowsTensor& x,
                                  const DenseTensor& scale,
                                  float bias,
                                  bool bias_after_scale,
                                  SelectedRowsTensor* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  Scale<T>(dev_ctx,
           x.value(),
           *scale.data<float>(),
           bias,
           bias_after_scale,
           out->mutable_value());
}

}  // namespace pt

using bfloat16 = ::paddle::platform::bfloat16;
PT_REGISTER_KERNEL("sign", CPU, NCHW, pt::Sign, float, double) {}
PT_REGISTER_KERNEL("mean", CPU, NCHW, pt::Mean, float, double) {}
PT_REGISTER_KERNEL("scale",
                   CPU,
                   NCHW,
                   pt::Scale,
                   float,
                   double,
                   bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.selectedrows",
                   CPU,
                   NCHW,
                   pt::ScaleSelectedRows,
                   float,
                   double,
                   bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
PT_REGISTER_KERNEL("scale.dynamic_attr",
                   CPU,
                   NCHW,
                   pt::ScaleDynamicAttr,
                   float,
                   double,
                   bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1)
      .SetBackend(pt::Backend::kCPU)
      .SetDataType(pt::DataType::kFLOAT32);
}
PT_REGISTER_KERNEL("scale.selectedrows.dynamic_attr",
                   CPU,
                   NCHW,
                   pt::ScaleSelectedRowsDynamicAttr,
                   float,
                   double,
                   bfloat16,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {
  kernel->InputAt(1)
      .SetBackend(pt::Backend::kCPU)
      .SetDataType(pt::DataType::kFLOAT32);
}

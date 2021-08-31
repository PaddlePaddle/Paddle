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

#include "paddle/top/cpu/math.h"

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

}  // namespace pt

using bfloat16 = ::paddle::platform::bfloat16;

// Register method 1:
// PT_REGISTER_KERNEL_STANDARD(sign, CPU, NCHW, FLOAT32, pt::Sign<float>)
//   .Input(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32))
//   .Output(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32));
// PT_TOUCH_KERNEL_REGISTRAR(sign, CPU, NCHW, FLOAT32);

// Register method 2:
// PT_REGISTER_KERNEL_AUTO_SPECIALIZE(sign, CPU, NCHW, pt::Sign, float)
//   .Input(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32))
//   .Output(BACKEND(CPU), DATALAYOUT(NCHW), DATATYPE(FLOAT32));
// PT_TOUCH_KERNEL_REGISTRAR(sign, CPU, NCHW, FLOAT32);

// Register method 3:
PT_REGISTER_KERNEL_2T(sign, CPU, NCHW, pt::Sign, float, double);
PT_REGISTER_KERNEL_2T(mean, CPU, NCHW, pt::Mean, float, double);
PT_REGISTER_KERNEL_8T(scale,
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
                      int64_t);

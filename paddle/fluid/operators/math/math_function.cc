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

#include "paddle/fluid/operators/math/math_function.h"

#ifdef PADDLE_WITH_MKLML
#include "paddle/fluid/platform/dynload/mklml.h"
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#endif

#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template struct SetConstant<platform::CPUDeviceContext, platform::float16>;
template struct SetConstant<platform::CPUDeviceContext, float>;
template struct SetConstant<platform::CPUDeviceContext, double>;
template struct SetConstant<platform::CPUDeviceContext, int>;
template struct SetConstant<platform::CPUDeviceContext, int64_t>;
template struct SetConstant<platform::CPUDeviceContext, bool>;
template struct SetConstant<platform::CPUDeviceContext, uint8_t>;

#define DEFINE_CPU_TRANS(RANK)                                             \
  template struct Transpose<platform::CPUDeviceContext, platform::float16, \
                            RANK>;                                         \
  template struct Transpose<platform::CPUDeviceContext, float, RANK>;      \
  template struct Transpose<platform::CPUDeviceContext, double, RANK>;     \
  template struct Transpose<platform::CPUDeviceContext, int, RANK>;        \
  template struct Transpose<platform::CPUDeviceContext, int64_t, RANK>;    \
  template struct Transpose<platform::CPUDeviceContext, bool, RANK>;       \
  template struct Transpose<platform::CPUDeviceContext, int16_t, RANK>;    \
  template struct Transpose<platform::CPUDeviceContext, uint8_t, RANK>;    \
  template struct Transpose<platform::CPUDeviceContext, int8_t, RANK>;

DEFINE_CPU_TRANS(1);
DEFINE_CPU_TRANS(2);
DEFINE_CPU_TRANS(3);
DEFINE_CPU_TRANS(4);
DEFINE_CPU_TRANS(5);
DEFINE_CPU_TRANS(6);

struct TensorSetConstantCPU {
  TensorSetConstantCPU(framework::Tensor* tensor, float value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    auto cpu = platform::CPUPlace();
    auto* begin = tensor_->mutable_data<T>(cpu);
    std::fill(begin, begin + tensor_->numel(), static_cast<T>(value_));
  }
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CPUPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(tensor->type(), TensorSetConstantCPU(tensor, value));
}

template <>
void set_constant_with_place<platform::CUDAPinnedPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(tensor->type(), TensorSetConstantCPU(tensor, value));
}

struct TensorSetConstantWithPlace : public boost::static_visitor<void> {
  TensorSetConstantWithPlace(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename Place>
  void operator()(Place place) const {
    set_constant_with_place<Place>(context_, tensor_, value_);
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value) {
  TensorSetConstantWithPlace func(context, tensor, value);
#ifdef PADDLE_WITH_CUDA
  tensor->place().apply_visitor(func);
#else
  func(platform::CPUPlace());
#endif
}

template <typename T>
struct RowwiseAdd<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(vector.numel(), size);
    PADDLE_ENFORCE_EQ(output->dims(), in_dims);

    auto in = framework::EigenMatrix<T>::From(input);
    auto vec = framework::EigenVector<T>::Flatten(vector);
    auto out = framework::EigenMatrix<T>::From(*output);

    for (int64_t i = 0; i < in_dims[0]; ++i) {
      out.chip(i, 0) = in.chip(i, 0) + vec;
    }
  }
};

template struct RowwiseAdd<platform::CPUDeviceContext, float>;
template struct RowwiseAdd<platform::CPUDeviceContext, double>;

template struct ColwiseSum<platform::CPUDeviceContext, float>;
template struct ColwiseSum<platform::CPUDeviceContext, double>;
template struct ColwiseSum<platform::CPUDeviceContext, int>;
template struct ColwiseSum<platform::CPUDeviceContext, int64_t>;

template struct RowwiseSum<platform::CPUDeviceContext, float>;
template struct RowwiseSum<platform::CPUDeviceContext, double>;

template struct RowwiseMean<platform::CPUDeviceContext, float>;
template struct RowwiseMean<platform::CPUDeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle

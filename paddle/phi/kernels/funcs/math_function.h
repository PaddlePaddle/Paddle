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

#pragma once
#include <cmath>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {
namespace funcs {

template <typename DeviceContext, typename T>
struct TransposeNormal {
  // for dims >= 7 situation
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& in,
                  paddle::framework::Tensor* out,
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& in,
                  paddle::framework::Tensor* out,
                  const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context,
                  paddle::framework::Tensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const paddle::platform::DeviceContext& context,
                             paddle::framework::Tensor* tensor,
                             float value);

void set_constant(const paddle::platform::DeviceContext& context,
                  paddle::framework::Tensor* tensor,
                  float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& input,
                  const paddle::framework::Tensor& vec,
                  paddle::framework::Tensor* output);
};

template <typename DeviceContext, typename T>
struct ElementwiseAddTo {
  // dst = dst + src
  void operator()(DeviceContext* ctx,
                  const paddle::framework::Tensor& src,
                  paddle::framework::Tensor* dst);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context,
                  const paddle::framework::Tensor& input,
                  paddle::framework::Tensor* vec);
};

#ifdef PADDLE_WITH_XPU
template <typename U>
struct TensorSetConstantXPU {
  TensorSetConstantXPU(paddle::framework::Tensor* tensor,
                       U value,
                       paddle::platform::Place place)
      : tensor_(tensor), value_(value), place_(place) {}
  template <typename T>
  void apply() const {
    auto* begin = tensor_->mutable_data<T>(place_);
    int numel = tensor_->numel();
    std::unique_ptr<T[]> data_cpu(new T[numel]);
    std::fill(data_cpu.get(), data_cpu.get() + numel, static_cast<T>(value_));
    paddle::memory::Copy(place_,
                         begin,
                         phi::CPUPlace(),
                         static_cast<void*>(data_cpu.get()),
                         numel * sizeof(T));
  }
  paddle::framework::Tensor* tensor_;
  U value_;
  paddle::platform::Place place_;
};
#endif

}  // namespace funcs
}  // namespace phi

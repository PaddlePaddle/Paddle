/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct DequantizeFunctor {
  void operator()(const DeviceContext& dev_ctx, const framework::Tensor* in,
                  const framework::Tensor* scale, float max_range,
                  framework::Tensor* out);
};

template <typename DeviceContext, typename T>
class DequantizeMaxAbsKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");

    auto* out = ctx.Output<framework::Tensor>("Out");

    float max_range = ctx.Attr<float>("max_range");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<float>(dev_ctx.GetPlace());

    DequantizeFunctor<DeviceContext, T>()(dev_ctx, in, scale, max_range, out);
  }
};

}  // namespace operators
}  // namespace paddle

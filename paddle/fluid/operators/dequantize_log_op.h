/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/ddim.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
struct DequantizeFunctor {
  void operator()(const DeviceContext& dev_ctx,
<<<<<<< HEAD
                  const framework::Tensor* in,
                  const framework::Tensor* dict,
                  framework::Tensor* out);
=======
                  const phi::DenseTensor* in,
                  const phi::DenseTensor* dict,
                  phi::DenseTensor* out);
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
};

template <typename DeviceContext, typename T>
class DequantizeLogKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<phi::DenseTensor>("X");
    auto* dict = ctx.Input<phi::DenseTensor>("Dict");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    out->mutable_data<float>(dev_ctx.GetPlace());

    DequantizeFunctor<DeviceContext, T>()(dev_ctx, in, dict, out);
  }
};

}  // namespace operators
}  // namespace paddle

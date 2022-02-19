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

#include <algorithm>
#include <utility>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

// only can include the headers in paddle/phi/include dirs
#include "paddle/phi/kernels/elementwise_grad_kernel.h"
#include "paddle/phi/kernels/math_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class ElementwiseAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<framework::LoDTensor>("X");
    auto *y = ctx.Input<framework::LoDTensor>("Y");
    auto *z = ctx.Output<framework::LoDTensor>("Out");
    z->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.device_context<DeviceContext>();
    int axis = ctx.Attr<int>("axis");
    pten::AddRawKernel<T>(
        static_cast<const typename framework::ConvertToPtenContext<
            DeviceContext>::TYPE &>(dev_ctx),
        *x, *y, axis, z);
  }
};

}  // namespace operators
}  // namespace paddle

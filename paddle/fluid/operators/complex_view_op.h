// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class AsComplexKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<framework::LoDTensor>("X");
    auto* out = context.Output<framework::LoDTensor>("Out");
    out->mutable_data<platform::complex<T>>(context.GetPlace());

    // TensorCopy also changes output's shape & dtype
    const framework::DDim out_dims_original = out->dims();
    framework::TensorCopy(*x, context.GetPlace(), out);
    out->Resize(out_dims_original);  // restored the shape
    out->mutable_data<platform::complex<T>>(
        context.GetPlace());  // restore the dtype
  }
};

template <typename DeviceContext, typename T>
class AsRealKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* x = context.Input<framework::LoDTensor>("X");
    auto* out = context.Output<framework::LoDTensor>("Out");

    out->mutable_data<T>(context.GetPlace());
    const framework::DDim out_dims_original = out->dims();
    framework::TensorCopy(*x, context.GetPlace(), out);
    out->Resize(out_dims_original);            // restored the shape
    out->mutable_data<T>(context.GetPlace());  // restore the dtype
  }
};

}  // namespace operators
}  // namespace paddle

// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

// DEPRECATED: The framework-level TensorContainsNAN/TensorContainsInf/
// TensorIsfinite functions have been moved to
// paddle/fluid/framework/tensor_isfinite.h.
// This header now only contains operators-level code.

#pragma once

#include "paddle/fluid/framework/tensor_isfinite.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

struct InfinityFunctor {
  void operator()(const phi::DenseTensor& tensor, phi::DenseTensor* out) {
    framework::TensorContainsInf(tensor, out);
  }
};

struct NANFunctor {
  void operator()(const phi::DenseTensor& tensor, phi::DenseTensor* out) {
    framework::TensorContainsNAN(tensor, out);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* x = ctx.InputVar("X");
    auto* out = ctx.Output<DenseTensor>("Out");
    out->template mutable_data<T>(ctx.GetPlace());
    Functor functor;
    if (x->IsType<DenseTensor>()) {
      auto* in = ctx.Input<DenseTensor>("X");
      functor(*in, out);
    } else if (x->IsType<phi::SelectedRows>()) {
      auto& in = ctx.Input<phi::SelectedRows>("X")->value();
      functor(in, out);
    } else {
      PADDLE_ENFORCE_EQ(true,
                        false,
                        common::errors::InvalidArgument(
                            "The input type mismatch, the type of Input(X) "
                            "must be DenseTensor or "
                            "SelectedRows, please check your input."));
    }
  }
};

}  // namespace operators
}  // namespace paddle

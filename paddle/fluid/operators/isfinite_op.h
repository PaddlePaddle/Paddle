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

#pragma once

#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/transform.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

struct InfinityFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsInf(tensor, out);
  }
};

struct NANFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorContainsNAN(tensor, out);
  }
};

struct IsfiniteFunctor {
  void operator()(const framework::Tensor& tensor, framework::Tensor* out) {
    framework::TensorIsfinite(tensor, out);
  }
};

template <typename DeviceContext, typename T, typename Functor>
class OverflowKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* x = ctx.InputVar("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    Functor functor;
    if (x->IsType<framework::LoDTensor>()) {
      auto* in = ctx.Input<framework::Tensor>("X");
      functor(*in, out);
    } else if (x->IsType<phi::SelectedRows>()) {
      auto& in = ctx.Input<phi::SelectedRows>("X")->value();
      functor(in, out);
    } else {
      PADDLE_ENFORCE_EQ(
          true, false,
          platform::errors::InvalidArgument(
              "The input type mismatch, the type of Input(X) must be Tensor or "
              "SelectedRows, please check your input."));
    }
  }
};

}  // namespace operators
}  // namespace paddle

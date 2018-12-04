/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"

namespace paddle {
namespace operators {

template <typename T>
struct MulFunctor {
  HOSTDEVICE T operator()(const T &a, const T &b) const { return a * b; }
};

template <typename DeviceContext, typename T>
class SelectedRowsMulTensorKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<framework::SelectedRows>("X");
    auto *y = context.Input<framework::Tensor>("Y");
    auto *z = context.Output<framework::SelectedRows>("Out");
    z->mutable_value()->mutable_data<T>(context.GetPlace());

    ElementwiseComputeEx<MulFunctor<T>, DeviceContext, T>(
        context, &(x->value()), y, -1 /*axis*/, MulFunctor<T>(),
        z->mutable_value());
  }
};

}  // namespace operators
}  // namespace paddle

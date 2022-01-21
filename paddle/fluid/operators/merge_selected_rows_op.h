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
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MergeSelectedRowsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<pten::SelectedRows>("X");
    auto* out = context.Output<pten::SelectedRows>("Out");

    math::scatter::MergeAdd<DeviceContext, T> merge_func;
    merge_func(context.template device_context<DeviceContext>(), *x, out);
  }
};

}  // namespace operators
}  // namespace paddle

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
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;

template <typename T>
class LookupSparseTableGradSplitKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const SelectedRows* in_grad = context.Input<SelectedRows>("Grad");

    auto in_rows = in_grad->rows();
    auto* out_row = context.Output<Tensor>("Row");
    out_row->Resize(
        framework::make_ddim({static_cast<int64_t>(in_rows.size()), 1}));
    out_row->mutable_data<int64_t>(context.GetPlace());
    framework::TensorFromVector(in_rows, context.device_context(), out_row);

    auto in_value = in_grad->value();
    std::vector<T> ins_vector;
    framework::TensorToVector(in_value, context.device_context(), &ins_vector);
    auto dims = in_value.dims();

    auto* out_v = context.OutputVar("Value");
    out_v->Clear();
    auto* out_t = out_v->GetMutable<framework::LoDTensor>();
    out_t->mutable_data<T>(context.GetPlace());
    framework::TensorFromVector(ins_vector, context.device_context(), out_t);
    out_t->Resize(dims);
  }
};
}  // namespace operators
}  // namespace paddle

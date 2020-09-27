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
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/large_scale_kv.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LargeScaleFuseSGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override;
};

template <typename T>
class LargeScaleFuseSGDOpKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *grad_var = ctx.InputVar("Grad");

    PADDLE_ENFORCE(
        grad_var->IsType<framework::SelectedRows>(),
        platform::errors::InvalidArgument(
            "in large scale optimize, gradient should only be SelectedRows"));

    const auto &grad = grad_var->Get<framework::SelectedRows>();

    // for distributed training, a sparse var may be empty,
    // just skip updating.
    if (grad.rows().size() == 0) {
      return;
    }

    framework::SelectedRows tmp_grad_merge;
    const framework::SelectedRows *grad_merge_ptr;
    math::scatter::MergeAdd<platform::CPUDeviceContext, T> merge_func;
    merge_func(ctx.template device_context<platform::CPUDeviceContext>(), grad,
               &tmp_grad_merge, true);
    grad_merge_ptr = &tmp_grad_merge;

    std::vector<int64_t> in_rows;
    in_rows.reserve(grad_merge_ptr->rows().size());
    std::copy(grad_merge_ptr->rows().begin(), grad_merge_ptr->rows().end(),
              std::back_inserter(in_rows));

    const auto *lr = learning_rate->data<T>();
    auto grad_v = grad_merge_ptr->value();
    auto grad_width = grad_v.dims()[1];

    //    auto is_entry = context.Attr<bool>("is_entry");
    auto tablename = ctx.Attr<std::string>("tablename");
    auto value_names = ctx.Attr<std::vector<std::string>>("value_names");

    std::vector<std::vector<std::vector<float> *>> values;
    std::vector<int64_t> dims;

    auto *ins = distributed::LargeScaleKV::GetInstance();
    auto *table = ins->Get(tablename);
    table->Get(in_rows, value_names, &values);
    table->Dims({"Param"}, &dims);

    PADDLE_ENFORCE_EQ(dims[0], grad_width,
                      platform::errors::InvalidArgument(
                          "param_row should have the same size with grad_row"));

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

    std::vector<T> grads;
    framework::TensorToVector(grad_v, ctx.device_context(), &grads);

    blas.SCAL(grads.size(), lr[0], grads.data());

    for (int x = 0; x < static_cast<int>(in_rows.size()); ++x) {
      auto &params = values[x][0];
      blas.VSUB(grad_width, params->data(), grads.data() + grad_width * x,
                params->data());
    }
  }
};
}  // namespace operators
}  // namespace paddle

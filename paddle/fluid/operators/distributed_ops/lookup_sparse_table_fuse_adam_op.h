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

#include <math.h>  // for sqrt in CPU and CUDA
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
class LargeScaleFuseAdamOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override;
};

template <typename T>
class LargeScaleFuseAdamOpKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    using paddle::framework::LoDTensor;

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

    auto *beta1_pow = ctx.Input<LoDTensor>("Beta1Pow");
    auto *beta2_pow = ctx.Input<LoDTensor>("Beta2Pow");
    auto *beta1_pow_out = ctx.Output<LoDTensor>("Beta1PowOut");
    auto *beta2_pow_out = ctx.Output<LoDTensor>("Beta2PowOut");
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
    T beta1 = static_cast<T>(ctx.Attr<float>("beta1"));
    T beta2 = static_cast<T>(ctx.Attr<float>("beta2"));

    PADDLE_ENFORCE_EQ(beta1_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta1 pow output size should be 1, but received "
                          "value is:%d.",
                          beta1_pow_out->numel()));

    PADDLE_ENFORCE_EQ(beta2_pow_out->numel(), 1,
                      platform::errors::InvalidArgument(
                          "beta2 pow output size should be 1, but received "
                          "value is:%d.",
                          beta2_pow_out->numel()));

    // update beta1 and beta2
    beta1_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
        beta1 * beta1_pow->data<T>()[0];
    beta2_pow_out->mutable_data<T>(ctx.GetPlace())[0] =
        beta2 * beta2_pow->data<T>()[0];

    std::vector<std::vector<std::vector<float> *>> values;
    std::vector<int64_t> dims;

    auto *ins = distributed::LargeScaleKV::GetInstance();
    auto *table = ins->Get(tablename);
    table->Get(in_rows, value_names, &values);
    table->Dims({"Param"}, &dims);

    PADDLE_ENFORCE_EQ(dims[0], grad_width,
                      platform::errors::InvalidArgument(
                          "param_row should have the same size with grad_row"));

    T lr_ = lr[0];
    T beta1_pow_ = beta1_pow->data<T>()[0];
    T beta2_pow_ = beta2_pow->data<T>()[0];

    lr_ *= sqrt(1 - beta2_pow_) / (1 - beta1_pow_);

    for (size_t i = 0; i < in_rows.size(); i++) {
      auto &params = values[i][0];
      auto &moment_1 = values[i][1];
      auto &moment_2 = values[i][2];

      auto *p_data = params->data();
      auto *m1_data = moment_1->data();
      auto *m2_data = moment_2->data();

      for (int x = 0; x < grad_width; ++x) {
        auto g = grad_v.data<T>()[grad_width * i + x];
        m1_data[x] = beta1 * m1_data[x] + (1 - beta1) * g;
        m2_data[x] = beta2 * m2_data[x] + (1 - beta2) * g * g;
        p_data[x] -= lr_ * (m1_data[x] / (sqrt(m2_data[x]) + epsilon));
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

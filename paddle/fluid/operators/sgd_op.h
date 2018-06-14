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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"

namespace paddle {
namespace operators {

template <typename T>
class SGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>()) {
      const auto *param = ctx.Input<framework::Tensor>("Param");
      auto *param_out = ctx.Output<framework::Tensor>("ParamOut");

      // Actually, all tensors are LoDTensor except SelectedRows.
      if (grad_var->IsType<framework::LoDTensor>()) {
        param_out->mutable_data<T>(ctx.GetPlace());
        const auto *grad = ctx.Input<framework::Tensor>("Grad");

        auto p = framework::EigenVector<T>::Flatten(*param);
        auto g = framework::EigenVector<T>::Flatten(*grad);
        auto o = framework::EigenVector<T>::Flatten(*param_out);
        auto *lr = learning_rate->data<T>();

        o = p - lr[0] * g;
      } else if (grad_var->IsType<framework::SelectedRows>()) {
        // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
        // This manual optimization brings difficulty to track data dependency.
        // It's better to find a more elegant solution.
        PADDLE_ENFORCE_EQ(param, param_out);
        const auto *grad = ctx.Input<framework::SelectedRows>("Grad");

        // for distributed training, a sparse var may be empty,
        // just skip updating.
        if (grad->rows().size() == 0) {
          return;
        }

        auto grad_height = grad->height();
        auto out_dims = param_out->dims();
        PADDLE_ENFORCE_EQ(grad_height, out_dims[0]);

        auto &grad_value = grad->value();
        auto &grad_rows = grad->rows();

        size_t grad_row_numel = grad_value.numel() / grad_rows.size();
        PADDLE_ENFORCE_EQ(static_cast<int64_t>(grad_row_numel),
                          param_out->numel() / grad_height);

        auto *grad_data = grad_value.data<T>();
        auto *out_data = param_out->data<T>();
        auto *lr = learning_rate->data<T>();
        for (size_t i = 0; i < grad_rows.size(); i++) {
          PADDLE_ENFORCE(grad_rows[i] < grad_height,
                         "Input rows index should less than height");
          for (size_t j = 0; j < grad_row_numel; j++) {
            out_data[grad_rows[i] * grad_row_numel + j] -=
                lr[0] * grad_data[i * grad_row_numel + j];
          }
        }
      } else {
        PADDLE_THROW("Unsupported Variable Type of Grad");
      }
    } else if (param_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(grad_var->IsType<framework::SelectedRows>(),
                     "when param "
                     "is SelectedRows, gradient should also be SelectedRows");
      const auto &param = param_var->Get<framework::SelectedRows>();
      auto *param_out = ctx.Output<framework::SelectedRows>("ParamOut");
      const auto &grad = grad_var->Get<framework::SelectedRows>();

      // for distributed training, a sparse var may be empty,
      // just skip updating.
      if (grad.rows().size() == 0) {
        return;
      }

      auto param_row_width = param.value().dims()[1];
      auto grad_row_width = grad.value().dims()[1];
      VLOG(4) << " param rows: " << param.rows().size()
              << " param memory rows: " << param.value().dims()[0]
              << " grad rows: " << grad.rows().size()
              << " grad memory rows: " << grad.value().dims()[0];
      PADDLE_ENFORCE_EQ(param_row_width, grad_row_width,
                        "param_row should have the same size with grad_row");

      const auto *lr = learning_rate->data<T>();
      const auto *grad_data = grad.value().data<T>();
      auto *out_data = param_out->mutable_value()->data<T>();
      for (size_t i = 0; i < grad.rows().size(); i++) {
        PADDLE_ENFORCE(grad.rows()[i] < grad.height(),
                       "Input rows index should less than height");
        int64_t id_index = param.Index(grad.rows()[i]);
        PADDLE_ENFORCE_GE(id_index, static_cast<int64_t>(0),
                          "id should be in the table");
        for (int64_t j = 0; j < grad_row_width; j++) {
          out_data[id_index * grad_row_width + j] -=
              lr[0] * grad_data[i * grad_row_width + j];
        }
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Parameter");
    }
  }
};
}  // namespace operators
}  // namespace paddle

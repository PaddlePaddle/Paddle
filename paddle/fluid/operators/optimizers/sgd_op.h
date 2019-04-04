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
#include "paddle/fluid/operators/jit/kernels.h"

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
        const auto *grad = ctx.Input<framework::Tensor>("Grad");
        auto sz = param_out->numel();
        PADDLE_ENFORCE_EQ(param->numel(), sz);
        PADDLE_ENFORCE_EQ(grad->numel(), sz);

        jit::sgd_attr_t attr(1, sz, 1, sz, 1);
        const T *lr = learning_rate->data<T>();
        const T *param_data = param->data<T>();
        const T *grad_data = grad->data<T>();
        int64_t rows_idx = 0;
        T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

        auto sgd =
            jit::Get<jit::kSgd, jit::SgdTuples<T>, platform::CPUPlace>(attr);
        sgd(lr, param_data, grad_data, &rows_idx, out_data, &attr);
      } else if (grad_var->IsType<framework::SelectedRows>()) {
        // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
        // This manual optimization brings difficulty to track data dependency.
        // It's better to find a more elegant solution.
        PADDLE_ENFORCE_EQ(param, param_out);
        const auto *grad = ctx.Input<framework::SelectedRows>("Grad");
        auto &grad_rows = grad->rows();

        // for distributed training, a sparse var may be empty,
        // just skip updating.
        if (grad_rows.size() == 0) {
          return;
        }

        auto out_dims = param_out->dims();
        PADDLE_ENFORCE_EQ(grad->height(), out_dims[0]);
        auto &grad_value = grad->value();
        const T *param_data = param->data<T>();
        const T *grad_data = grad_value.data<T>();
        const T *lr = learning_rate->data<T>();
        const int64_t *rows_data = grad_rows.data();
        T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

        jit::sgd_attr_t attr;
        attr.param_height = out_dims[0];
        attr.param_width = param_out->numel() / attr.param_height;
        attr.grad_height = grad_rows.size();  // note: it is not grad->height()
        attr.grad_width = grad_value.numel() / attr.grad_height;
        attr.selected_rows_size = grad_rows.size();
        PADDLE_ENFORCE_EQ(attr.grad_width, attr.param_width);

        if (attr.grad_width  == 1) {
          VLOG(1) << "width ==  1";
          for (int i = 0; i < grad_rows.size(); i++) {
            if(rows_data[i] == 0) {
               VLOG(2) << "update row 0";
            }
            VLOG(1) << "update row " << rows_data[i];
          }
        }

        auto sgd =
            jit::Get<jit::kSgd, jit::SgdTuples<T>, platform::CPUPlace>(attr);
        sgd(lr, param_data, grad_data, rows_data, out_data, &attr);
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
        int64_t id_index = param_out->AutoGrownIndex(grad.rows()[i], false);
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

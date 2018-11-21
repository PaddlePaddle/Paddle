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
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/memory/memcpy.h"

namespace paddle {
namespace operators {

template <typename T>
class SparseParameterExtractOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>());
    PADDLE_ENFORCE(grad_var->IsType<framework::SelectedRows>());

    const auto *param = ctx.Input<framework::Tensor>("Param");
    const auto *grad = ctx.Input<framework::SelectedRows>("Grad");

    auto *param_out = ctx.Output<framework::SelectedRows>("ParamOut");

    // for distributed training, a sparse var may be empty,
    // just skip updating.
    if (grad->rows().empty()) {
      return;
    }

    auto &grad_value = grad->value();
    auto &grad_rows = grad->rows();

    size_t grad_row_numel = grad_value.numel() / grad_rows.size();

    param_out->set_height(grad->height());
    param_out->set_rows(grad_rows);
    auto *param_out_data = param_out->mutable_value()->mutable_data<T>(
        grad->value().dims(), ctx.GetPlace());

    auto grad_height = grad->height();

    auto *param_data = param->data<T>();
    for (size_t i = 0; i < grad_rows.size(); i++) {
      PADDLE_ENFORCE(grad_rows[i] < grad_height,
                     "Input rows index should less than height");
      auto place = ctx.GetPlace();
      if (platform::is_cpu_place(place)) {
        auto ctx_cpu_place = boost::get<platform::CPUPlace>(place);
        memory::Copy(ctx_cpu_place, param_out_data + i * grad_row_numel,
                     ctx_cpu_place, param_data + grad_rows[i] * grad_row_numel,
                     sizeof(T) * grad_row_numel);
      }
#ifdef PADDLE_WITH_CUDA
      else if (platform::is_gpu_place(place)) {  // NOLINT
        auto ctx_gpu_place = boost::get<platform::CUDAPlace>(place);
        auto stream =
            reinterpret_cast<const platform::CUDADeviceContext &>(ctx).stream();
        memory::Copy(ctx_gpu_place, dst_ptr, ctx_gpu_place, src_ptr, size,
                     stream);
      }
#endif
    }
  }
};
}  // namespace operators
}  // namespace paddle

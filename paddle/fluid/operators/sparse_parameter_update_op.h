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
class SparseParameterUpdateOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *param_var = ctx.InputVar("Param");
    const auto *SparseParam_var = ctx.InputVar("SparseParam");

    PADDLE_ENFORCE(param_var->IsType<framework::LoDTensor>());
    PADDLE_ENFORCE(SparseParam_var->IsType<framework::SelectedRows>());

    const auto *param = ctx.Input<framework::Tensor>("Param");
    const auto *SparseParam = ctx.Input<framework::SelectedRows>("SparseParam");

    auto *param_out = ctx.Output<framework::SelectedRows>("ParamOut");

    // for distributed training, a sparse var may be empty,
    // just skip updating.
    if (SparseParam->rows().size() == 0) {
      return;
    }

    auto &SparseParam_value = SparseParam->value();
    auto &SparseParam_rows = SparseParam->rows();

    size_t SparseParam_row_numel =
        SparseParam_value.numel() / SparseParam_rows.size();

    param_out->set_height(SparseParam->height());
    param_out->set_rows(SparseParam_rows);
    auto *param_out_data = param_out->mutable_value()->mutable_data<T>(
        SparseParam->value().dims(), ctx.GetPlace());

    auto SparseParam_height = SparseParam->height();

    auto *param_data = param->data<T>();
    for (size_t i = 0; i < SparseParam_rows.size(); i++) {
      PADDLE_ENFORCE(SparseParam_rows[i] < SparseParam_height,
                     "Input rows index should less than height");
      for (size_t j = 0; j < SparseParam_row_numel; j++) {
        auto place = ctx.GetPlace();
        if (platform::is_cpu_place(place)) {
          auto ctx_cpu_place = boost::get<platform::CPUPlace>(place);
          memory::Copy(ctx_cpu_place,
                       param_out_data + i * SparseParam_row_numel,
                       ctx_cpu_place,
                       param_data + SparseParam_rows[i] * SparseParam_row_numel,
                       sizeof(T) * SparseParam_row_numel);
        }
#ifdef PADDLE_WITH_CUDA
        else if (platform::is_gpu_place(place)) {  // NOLINT
          auto ctx_gpu_place = boost::get<platform::CUDAPlace>(place);
          auto stream =
              reinterpret_cast<const platform::CUDADeviceContext &>(ctx)
                  .stream();
          memory::Copy(ctx_gpu_place, dst_ptr, ctx_gpu_place, src_ptr, size,
                       stream);
        }
#endif
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle

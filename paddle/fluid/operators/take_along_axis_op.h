/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/gather_scatter_kernel.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class TakeAlongAxisOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input = ctx.Input<Tensor>("Input");
    VLOG(3) << " input:" << *(input->data<T>());
    auto dim = ctx.Attr<int>("Dim");
    auto index = ctx.Input<Tensor>("Index");
    auto result = ctx.Output<Tensor>("Result");
    result->Resize(index->dims());
    result->mutable_data<T>(ctx.GetPlace());
    // // resize_output(result, index.sizes());
    // // check_no_internal_overlap(self, result);
    // // check_no_partial_overlap(result, index);
    // VLOG(3) << "000000000";

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      cpu_gather_kernel<T, int32_t>(*input, dim, *index, *result);
    } else if (index_type == framework::proto::VarType::INT64) {
      cpu_gather_kernel<T, int64_t>(*input, dim, *index, *result);
    }
    VLOG(3) << "<<<< Done Compute <<<<<";
  }
};

template <typename T>
class TakeAlongAxisGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("This kernel only runs on CPU."));

    auto input_d = ctx.Output<Tensor>("Input");
    VLOG(3) << " input:" << *(input_d->data<T>());
    auto index = ctx.Input<Tensor>("Index");
    auto result_d = ctx.Input<Tensor>("Result");

    auto dim = ctx.Attr<int>("Dim");
    input_d->Resize(index->dims());
    input_d->mutable_data<T>(ctx.GetPlace());
    // // resize_output(result, index.sizes());
    // // check_no_internal_overlap(self, result);
    // // check_no_partial_overlap(result, index);
    // VLOG(3) << "000000000";

    const auto &index_type = index->type();
    if (index_type == framework::proto::VarType::INT32) {
      cpu_scatter_add_kernel<T, int32_t>(
          *input_d, dim, *index,
          *result_d);  // the gradient of gather is scatter
    } else if (index_type == framework::proto::VarType::INT64) {
      cpu_scatter_add_kernel<T, int64_t>(*input_d, dim, *index, *result_d);
    }
    VLOG(3) << "<<<< Done Compute <<<<<";
  }
};

}  // namespace operators
}  // namespace paddle

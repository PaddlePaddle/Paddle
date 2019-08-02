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
#include "paddle/fluid/operators/scatter_nd.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class ScatterNDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LOG(ERROR) << 1;
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU.");
    LOG(ERROR) << 1;
    auto *X = ctx.Input<Tensor>("X");
    LOG(ERROR) << 1;
    auto *Ids = ctx.Input<Tensor>("Ids");
    LOG(ERROR) << 1;
    auto *Updates = ctx.Input<Tensor>("Updates");
    LOG(ERROR) << 1;
    auto *Out = ctx.Output<Tensor>("Out");
    LOG(ERROR) << 1;
    int dim = ctx.Attr<int>("dim");
    LOG(ERROR) << 1;

    // In place output: Out = X, Out[Ids] = Updates
    framework::TensorCopySync(*X, ctx.GetPlace(), Out);
    LOG(ERROR) << 1;
    // Apply ScatterUpdate: Out[index] = Updates[:]
    const auto &index_type = Ids->type();
    LOG(ERROR) << 1;
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    LOG(ERROR) << 1;
    PADDLE_ENFORCE(
        index_type_match,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(index_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    LOG(ERROR) << 1;
    if (index_type == framework::proto::VarType::INT32) {
      LOG(ERROR) << 1;
      ScatterNDAssign<T, int32_t>(ctx.device_context(), *Updates, *Ids, Out,
                                  dim);
    } else {
      LOG(ERROR) << 1;
      ScatterNDAssign<T, int64_t>(ctx.device_context(), *Updates, *Ids, Out,
                                  dim);
    }
    LOG(ERROR) << 1;
  }
};

}  // namespace operators
}  // namespace paddle

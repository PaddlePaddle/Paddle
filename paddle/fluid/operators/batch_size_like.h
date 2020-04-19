/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

class BatchSizeLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of %s should not be null.", Type());
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of %s should not be null.", Type());

    auto &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE_GT(shape.size(), 0);
    std::vector<int64_t> shape_int64(shape.size(), 0);
    std::transform(shape.begin(), shape.end(), shape_int64.begin(),
                   [](int a) { return static_cast<int64_t>(a); });
    auto output_dim = framework::make_ddim(shape_int64);

    int input_dim_idx = ctx->Attrs().Get<int>("input_dim_idx");
    PADDLE_ENFORCE_GE(input_dim_idx, 0);
    PADDLE_ENFORCE_GT(ctx->GetInputDim("Input").size(), input_dim_idx);

    int output_dim_idx = ctx->Attrs().Get<int>("output_dim_idx");
    PADDLE_ENFORCE_GE(output_dim_idx, 0);
    PADDLE_ENFORCE_GT(static_cast<int>(shape.size()), output_dim_idx);

    output_dim[output_dim_idx] = ctx->GetInputDim("Input")[input_dim_idx];
    ctx->SetOutputDim("Out", output_dim);
  }
};

class BatchSizeLikeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput(
        "Input",
        "Tensor whose input_dim_idx'th dimension specifies the batch_size");
    AddOutput("Out",
              "Tensor of specified shape will be filled "
              "with the specified value");
    AddAttr<std::vector<int>>("shape", "The shape of the output");
    AddAttr<int>("input_dim_idx",
                 "default 0. The index of input's batch size dimension")
        .SetDefault(0);
    AddAttr<int>("output_dim_idx",
                 "default 0. The index of output's batch size dimension")
        .SetDefault(0);
    Apply();
  }

 protected:
  virtual void Apply() = 0;
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(BatchSizeLikeNoNeedBufferVarsInference,
                                    "Input");

}  // namespace operators
}  // namespace paddle

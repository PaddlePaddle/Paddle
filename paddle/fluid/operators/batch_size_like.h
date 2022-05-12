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
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using MetaTensor = framework::CompatMetaTensor;

class BatchSizeLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", Type());
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", Type());

    MetaTensor x(ctx->GetInputVarPtrs("Input")[0], ctx->IsRuntime());
    MetaTensor out(ctx->GetOutputVarPtrs("Out")[0], ctx->IsRuntime());
    auto& shape = ctx->Attrs().Get<std::vector<int>>("shape");
    int x_batch_size_dim = ctx->Attrs().Get<int>("input_dim_idx");
    int out_batch_size_dim = ctx->Attrs().Get<int>("output_dim_idx");
    phi::BatchSizeLikeInferMeta(x, shape, x_batch_size_dim, out_batch_size_dim,
                                &out);
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

DECLARE_NO_NEED_BUFFER_VARS_INFERER(BatchSizeLikeNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

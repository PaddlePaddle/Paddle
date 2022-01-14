/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

class FusedGemmEpilogueOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(ctx->HasInput("bias"), "Output", "bias",
                   "FusedGemmEpilogueOp");
    OP_INOUT_CHECK(ctx->HasOutput("out"), "Output", "out",
                   "FusedGemmEpilogueOp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto bias_dims = ctx->GetInputDim("bias");

    auto trans_x = ctx->Attrs().Get<bool>("trans_x");
    auto trans_y = ctx->Attrs().Get<bool>("trans_y");

    PADDLE_ENFORCE_EQ(
        y_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The Input tensor Y's dimension of FusedGemmEpilogueOp "
            " should be 2, but got %d.",
            y_dims.size()));

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The Input tensor X's dimension of FusedGemmEpilogueOp "
            " should be >= 2, but got %d.",
            x_dims.size()));

    PADDLE_ENFORCE_EQ(
        bias_dims.size(), 1,
        platform::errors::InvalidArgument(
            "The Input tensor bias's dimension of FusedGemmEpilogueOp "
            " should be == 1, but got %d.",
            bias_dims.size()));

    PADDLE_ENFORCE_EQ(bias_dims[0], trans_y ? y_dims[0] : y_dims[1],
                      platform::errors::InvalidArgument(
                          "The Input tensor bias's dimension 0"
                          " should be == Y[-1], but got bias's shape = [%s] "
                          "and Y's shape = [%s]",
                          bias_dims, y_dims));

    auto x_mat_dims =
        framework::flatten_to_2d(x_dims, trans_x ? 1 : x_dims.size() - 1);

    int K_from_x = trans_x ? x_mat_dims[0] : x_mat_dims[1];
    int K_from_y = trans_y ? y_dims[1] : y_dims[0];

    PADDLE_ENFORCE_EQ(
        K_from_x, K_from_y,
        platform::errors::InvalidArgument(
            "The last dimension of X should be equal with Y's first dimension."
            "But received X[-1] = [%d], Y[0] = [%d].",
            K_from_x, K_from_y));

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<size_t>(x_dims.size()));
    if (trans_x) {
      for (int i = 1; i < x_dims.size(); ++i) out_dims.push_back(x_dims[i]);
    } else {
      for (int i = 0; i < x_dims.size() - 1; ++i) out_dims.push_back(x_dims[i]);
    }

    if (trans_y)
      out_dims.push_back(y_dims[0]);
    else
      out_dims.push_back(y_dims[1]);

    ctx->SetOutputDim("out", framework::make_ddim(out_dims));
    if (ctx->HasOutput("auxiliary")) {
      ctx->SetOutputDim("out", framework::make_ddim(out_dims));
    }
  }
};

class FusedGemmEpilogueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor X of Out = (X * Y) + bias.");
    AddInput("Y", "The input tensor X of Out = (X * Y) + bias.");
    AddInput("bias", "The input tensor bias of Out = (X * Y) + bias.");

    AddOutput("out", "The output tensor X of Out = (X * Y) + bias.");
    AddOutput("auxiliary",
              "The Output tensor of auxiliary data."
              "For ReLU, it should be bits in `out` shape, "
              "and For GeLU should be <T> in `out` shape.")
        .AsDispensable();

    AddAttr<bool>("trans_x", "").SetDefault(false);
    AddAttr<bool>("trans_y", "").SetDefault(false);

    AddComment(R"DOC(FusedGemmEpilogue OP)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_gemm_epilogue, ops::FusedGemmEpilogueOp,
                  ops::FusedGemmEpilogueOpMaker)

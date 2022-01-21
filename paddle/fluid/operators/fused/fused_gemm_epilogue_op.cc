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

#include "paddle/fluid/operators/fused/fused_gemm_epilogue_op.h"

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

    auto activation = ctx->Attrs().Get<std::string>("activation");
    auto auxiliary_key = ctx->Attrs().Get<std::string>("auxiliary_key");

    if ((activation != "relu") && (activation != "gelu") &&
        (activation != "none")) {
      PADDLE_ENFORCE_EQ(
          true, false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation));
    }

    if (activation == "none" && auxiliary_key.size() > 0) {
      VLOG(0)
          << "[Warning] auxiliary_key:The auxiliary_key would not be used when "
          << "activation = \"none\"";
    }
    // cublasLt's restriction for auxiliary.
    if (auxiliary_key.size() > 0 && activation != "none") {
      int min_size_of_n = activation == "relu" ? 128 : 8;
      int N_size = trans_y ? y_dims[0] : y_dims[1];
      PADDLE_ENFORCE_EQ(N_size % min_size_of_n, 0,
                        platform::errors::InvalidArgument(
                            "The output dimension N (X(MxK) * Y(KxN) = C(MxN)) "
                            "should be multiple of %d when auxiliary_key given "
                            "and activation=%s, but got N = %d.",
                            min_size_of_n, activation, N_size));
    }

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
  }
};

class FusedGemmEpilogueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor X of Out = (X * Y) + bias.");
    AddInput("Y", "The input tensor X of Out = (X * Y) + bias.");
    AddInput("bias", "The input tensor bias of Out = (X * Y) + bias.");

    AddOutput("out", "The output tensor X of Out = (X * Y) + bias.");

    AddAttr<bool>("trans_x", "").SetDefault(false);
    AddAttr<bool>("trans_y", "").SetDefault(false);

    AddAttr<std::string>("activation", "{noen, relu, gelu}").SetDefault("none");
    AddAttr<std::string>("auxiliary_key", "").SetDefault("");

    AddComment(R"DOC(FusedGemmEpilogue OP)DOC");
  }
};

class FusedGemmEpilogueGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("DOut"), "Input", "DOut",
                   "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "FusedGemmEpilogueGradOp");
    OP_INOUT_CHECK(ctx->HasOutput("DY"), "Output", "DY", "FusedGemmEpilogueOp");

    auto dout_dims = ctx->GetInputDim("DOut");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_GE(
        dout_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The Input tensor DOut's dimension of FusedGemmEpilogueGradOp "
            " should be >= 2, but got %d.",
            dout_dims.size()));

    PADDLE_ENFORCE_EQ(
        y_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The Input tensor Y's dimension of FusedGemmEpilogueGradOp "
            " should be 2, but got %d.",
            y_dims.size()));

    PADDLE_ENFORCE_GE(
        x_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The Input tensor X's dimension of FusedGemmEpilogueGradOp "
            " should be >= 2, but got %d.",
            x_dims.size()));

    PADDLE_ENFORCE_EQ(
        dout_dims.size(), x_dims.size(),
        platform::errors::InvalidArgument(
            "The Input tensor DOut's and X's dimension of "
            "FusedGemmEpilogueGradOp "
            " should be the same, but got DOut's dim = %d and X's = %d.",
            dout_dims.size(), x_dims.size()));

    auto dout_mat_dims =
        framework::flatten_to_2d(dout_dims, dout_dims.size() - 1);

    auto x_mat_dims = framework::flatten_to_2d(x_dims, x_dims.size() - 1);

    PADDLE_ENFORCE_EQ(
        dout_mat_dims[1], y_dims[1],
        platform::errors::InvalidArgument(
            "The last dimension of DOut should be equal with Y's last"
            "dimension. But received DOut[-1] = [%d], Y[1] = [%d].",
            dout_mat_dims[1], y_dims[1]));

    PADDLE_ENFORCE_EQ(
        dout_mat_dims[0], x_mat_dims[0],
        platform::errors::InvalidArgument(
            "The first dimension of DOut should be equal with X's first"
            "dimension. But received DOut[0] = [%d], Y[0] = [%d].",
            dout_mat_dims[0], x_mat_dims[0]));

    auto activation_grad = ctx->Attrs().Get<std::string>("activation_grad");
    auto auxiliary_key = ctx->Attrs().Get<std::string>("auxiliary_key");
    if ((activation_grad != "relu_grad") && (activation_grad != "gelu_grad") &&
        (activation_grad != "none")) {
      PADDLE_ENFORCE_EQ(
          true, false,
          platform::errors::InvalidArgument(
              "The activation attribute of fused_gemm_epilogue op should be"
              " one of {\"none\", \"relu\", \"gelu\"}. But received %s."
              "But received activation=%s.",
              activation_grad));
    }

    if (activation_grad != "none" && auxiliary_key.size() == 0) {
      PADDLE_ENFORCE_EQ(true, false,
                        platform::errors::InvalidArgument(
                            "The auxiliary_key should not be empty string. "
                            "when activation_grad == {relu_grad, gelu_grad}."));
    }

    if (ctx->HasOutput("DX")) {
      std::vector<int64_t> dx_dims;
      dx_dims.reserve(static_cast<size_t>(x_dims.size()));
      for (int i = 0; i < x_dims.size(); ++i) {
        dx_dims.push_back(x_dims[i]);
      }
      ctx->SetOutputDim("DX", framework::make_ddim(dx_dims));
    }

    std::vector<int64_t> dy_dims;
    dy_dims.reserve(static_cast<size_t>(y_dims.size()));
    for (int i = 0; i < y_dims.size(); ++i) {
      dy_dims.push_back(y_dims[i]);
    }
    ctx->SetOutputDim("DY", framework::make_ddim(dy_dims));

    if (ctx->HasOutput("DBias")) {
      std::vector<int64_t> dbias_dims;
      dbias_dims.reserve(1);
      dbias_dims.push_back(y_dims[1]);
      ctx->SetOutputDim("DBias", framework::make_ddim(dbias_dims));
    }
  }
};

class FusedGemmEpilogueGradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("DOut", "The input grad tensor DOut of  Out = (X * Y) + bias.");
    AddInput("X", "The input tensor X of Out = (X * Y) + bias.");
    AddInput("Y", "The input tensor X of Out = (X * Y) + bias.");

    AddOutput("DX", "The output grad tensor DX of Out = (X * Y) + bias.")
        .AsDispensable();
    AddOutput("DY", "The output grad tensor DY of Out = (X * Y) + bias.");
    AddOutput("DBias", "The output grad tensor DBias of Out = (X * Y) + bias.")
        .AsDispensable();

    AddAttr<std::string>("activation_grad", "{noen, relu_grad, gelu_grad}")
        .SetDefault("none");
    AddAttr<std::string>("auxiliary_key", "").SetDefault("");

    AddComment(R"DOC(FusedGemmEpilogueGrad OP)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_gemm_epilogue, ops::FusedGemmEpilogueOp,
                  ops::FusedGemmEpilogueOpMaker)
REGISTER_OPERATOR(fused_gemm_epilogue_grad, ops::FusedGemmEpilogueGradOp,
                  ops::FusedGemmEpilogueGradOpMaker)

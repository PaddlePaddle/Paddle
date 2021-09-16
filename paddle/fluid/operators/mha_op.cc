/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
Copyright (c) 2021 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/mha_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MHAOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Q"), "Input", "Q", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("K"), "Input", "K", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("V"), "Input", "V", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("QO_Seqlen"), "Input", "QO_Seqlen", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("KV_Seqlen"), "Input", "KV_Seqlen", "MHA");

    auto q_dims = ctx->GetInputDim("Q");
    PADDLE_ENFORCE_EQ(q_dims.size(), CUDNN_SEQDATA_DIM_COUNT,
                      platform::errors::InvalidArgument(
                          "The input tensor Q's dimensions of MHAOp "
                          "should be equal to %d . But received Q's "
                          "dimensions = %d.",
                          CUDNN_SEQDATA_DIM_COUNT, q_dims.size()));

    auto k_dims = ctx->GetInputDim("K");
    PADDLE_ENFORCE_EQ(k_dims.size(), CUDNN_SEQDATA_DIM_COUNT,
                      platform::errors::InvalidArgument(
                          "The input tensor K's dimensions of MHAOp "
                          "should be equal to %d . But received K's "
                          "dimensions = %d.",
                          CUDNN_SEQDATA_DIM_COUNT, k_dims.size()));

    auto v_dims = ctx->GetInputDim("V");
    PADDLE_ENFORCE_EQ(v_dims.size(), CUDNN_SEQDATA_DIM_COUNT,
                      platform::errors::InvalidArgument(
                          "The input tensor V's dimensions of MHAOp "
                          "should be equal to %d . But received V's "
                          "dimensions = %d.",
                          CUDNN_SEQDATA_DIM_COUNT, v_dims.size()));

    auto qo_slen_dims = ctx->GetInputDim("QO_Seqlen");
    PADDLE_ENFORCE_EQ(qo_slen_dims[0], q_dims[0],
                      platform::errors::InvalidArgument(
                          "The number of sequence length should be equal"
                          " to batch size."));

    auto kv_slen_dims = ctx->GetInputDim("KV_Seqlen");
    PADDLE_ENFORCE_EQ(kv_slen_dims[0], k_dims[0],
                      platform::errors::InvalidArgument(
                          "The number of sequence length should be equal"
                          " to batch size."));

    std::vector<int64_t> output_dims;
    for (int i = 0; i < q_dims.size(); ++i) {
      output_dims.push_back(q_dims[i]);
    }

    ctx->SetOutputDim("O", framework::make_ddim(output_dims));
    ctx->ShareLoD("Q", /*->*/ "O");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Q");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

class MHAOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "(Tensor), Q");
    AddInput("K", "(Tensor), K");
    AddInput("V", "(Tensor), V");
    AddInput("W", "(Tensor), V");
    AddInput("QO_Seqlen", "(Tensor), QO_Seqlen");
    AddInput("KV_Seqlen", "(Tensor), KV_Seqlen");

    AddOutput("O", "(Tensor), O");

    AddAttr<std::vector<int>>("attn_low_windows", "(Tensor), attn_low_windows");
    AddAttr<std::vector<int>>("attn_high_windows",
                              "(Tensor), attn_high_windows");

    AddAttr<float>("attn_dropout_rate", "");
    AddAttr<int>("attn_heads", "");
    AddAttr<float>("attn_sm_scaler", "");
    AddAttr<int>("attn_vec_size", "");
    AddAttr<int>("attn_q_proj_size", "");
    AddAttr<int>("attn_k_proj_size", "");
    AddAttr<int>("attn_v_proj_size", "");
    AddAttr<int>("attn_o_proj_size", "");
    AddAttr<int>("attn_max_qo_seq_len", "");
    AddAttr<int>("attn_max_kv_seq_len", "");
    AddAttr<int>("attn_beam_size", "");

    AddComment(R"DOC(MHA OP Test)DOC");
  }
};

class MHAOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"Q", /*->*/ "O"}};
    return m;
  }
};

class MHAGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("O")), "Input",
                   "O@GRAD", "mha");
    OP_INOUT_CHECK(ctx->HasInput("QO_Seqlen"), "Input", "QO_Seqlen", "mha");
    OP_INOUT_CHECK(ctx->HasInput("KV_Seqlen"), "Input", "KV_Seqlen", "mha");

    std::string var_names[4] = {"Q", "K", "V", "W"};
    for (auto s : var_names) {
      OP_INOUT_CHECK(ctx->HasInput(s), "Input", s, "mha");
      auto dims = ctx->GetInputDim(s);
      auto grad_name = framework::GradVarName(s);

      if (ctx->HasOutput(grad_name)) {
        ctx->SetOutputDim(grad_name, dims);
      }
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Q");
    return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
  }
};

template <typename T>
class MHAOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("mha_grad");
    retv->SetInput("Q", this->Input("Q"));
    retv->SetInput("K", this->Input("K"));
    retv->SetInput("V", this->Input("V"));
    retv->SetInput("W", this->Input("W"));
    retv->SetInput("QO_Seqlen", this->Input("QO_Seqlen"));
    retv->SetInput("KV_Seqlen", this->Input("KV_Seqlen"));
    retv->SetInput(framework::GradVarName("O"), this->OutputGrad("O"));
    retv->SetOutput(framework::GradVarName("Q"), this->InputGrad("Q"));
    retv->SetOutput(framework::GradVarName("K"), this->InputGrad("K"));
    retv->SetOutput(framework::GradVarName("V"), this->InputGrad("V"));
    retv->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(mha, ops::MHAOp, ops::MHAOpMaker, ops::MHAOpInferVarType,
                  ops::MHAOpGradMaker<paddle::framework::OpDesc>,
                  ops::MHAOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(mha_grad, ops::MHAGradOp);

namespace plat = paddle::platform;

REGISTER_OP_CPU_KERNEL(
    mha, ops::MHAKernel<paddle::platform::CPUDeviceContext, plat::float16>,
    ops::MHAKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MHAKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    mha_grad,
    ops::MHAGradKernel<paddle::platform::CPUDeviceContext, plat::float16>,
    ops::MHAGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::MHAGradKernel<paddle::platform::CPUDeviceContext, double>);

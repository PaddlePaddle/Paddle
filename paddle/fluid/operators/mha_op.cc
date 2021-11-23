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
    OP_INOUT_CHECK(ctx->HasInput("QO_KV_Seqlen"), "Input", "QO_KV_Seqlen", "MHA");

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

    auto qo_kv_slen_dims = ctx->GetInputDim("QO_KV_Seqlen");
    if (ctx->IsRuntime()) { 
      PADDLE_ENFORCE_EQ(qo_kv_slen_dims[0], 2*q_dims[0],
                        platform::errors::InvalidArgument(
                            "The number of sequence length should be equal"
                            " to 2*(batch size)."));
    }

    if (ctx->HasInput("low_high_windows")) {
      auto low_windows = ctx->GetInputDim("low_high_windows");
      if (ctx->IsRuntime()) { 
        PADDLE_ENFORCE_EQ(low_windows[0], 2*q_dims[1],
                          platform::errors::InvalidArgument(
                              "The number of low_high_windows should be equal"
                              " to 2*(sequence length)."));
      }
    }

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

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                    expected_kernel_type.place_,
                                    tensor.layout());
  }
};

class MHAOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Q", "(Tensor), Q");
    AddInput("K", "(Tensor), K");
    AddInput("V", "(Tensor), V");
    AddInput("W", "(Tensor), V");
    AddInput("QO_KV_Seqlen", "(Tensor), QO_KV_Seqlen");
    AddInput("low_high_windows", "(Tensor), low_windows and high_windows").AsDispensable();
    // This is for connecting computing graphs with MHA_SEQ_DATA_Prep Op when converting dygraph to static.
    // Since to_static would build ParallelExecutor which would run ops async if there is 
    // no dependence. Moreover, static.save_inference_model would prune graphs. If the nodes is 
    // not related the data flow from inputs to outputs, it would be removed.
    AddInput("fake_input", "(bool)").AsDispensable();

    AddOutput("O", "(Tensor), O");

    AddAttr<std::string>("cache_key", "");
    AddAttr<std::string>("seq_data_key", "").SetDefault("");

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
    OP_INOUT_CHECK(ctx->HasInput("QO_KV_Seqlen"), "Input", "QO_Seqlen", "mha");


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

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                    expected_kernel_type.place_,
                                    tensor.layout());
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
    retv->SetInput("QO_KV_Seqlen", this->Input("QO_KV_Seqlen"));
    if (this->HasInput("low_high_windows")) {
      retv->SetInput("low_high_windows", this->Input("low_high_windows"));
    }

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

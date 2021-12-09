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

#include "paddle/fluid/operators/mha/mha_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class MHAOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("query"), "Input", "query", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("key"), "Input", "key", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("value"), "Input", "value", "MHA");
    OP_INOUT_CHECK(ctx->HasInput("qo_kv_seqlen"), "Input", "qo_kv_seqlen",
                   "MHA");

    // CUDNN_SEQDATA_DIM_COUNT = 4, mins 1 = 3
    // due to omit beam_dim currently to have
    // a same inputs with Paddle's original MHA Layer.
    int qkv_dims_size = CUDNN_SEQDATA_DIM_COUNT - 1;
    auto query_dims = ctx->GetInputDim("query");
    PADDLE_ENFORCE_EQ(query_dims.size(), qkv_dims_size,
                      platform::errors::InvalidArgument(
                          "The input tensor query's dimensions of MHAOp "
                          "should be equal to %d . But received query's "
                          "dimensions = %d.",
                          qkv_dims_size, query_dims.size()));

    auto key_dims = ctx->GetInputDim("key");
    PADDLE_ENFORCE_EQ(key_dims.size(), qkv_dims_size,
                      platform::errors::InvalidArgument(
                          "The input tensor key's dimensions of MHAOp "
                          "should be equal to %d . But received key's "
                          "dimensions = %d.",
                          qkv_dims_size, key_dims.size()));

    auto value_dims = ctx->GetInputDim("value");
    PADDLE_ENFORCE_EQ(value_dims.size(), qkv_dims_size,
                      platform::errors::InvalidArgument(
                          "The input tensor value's dimensions of MHAOp "
                          "should be equal to %d . But received value's "
                          "dimensions = %d.",
                          qkv_dims_size, value_dims.size()));

    bool enable_bias = ctx->Attrs().Get<bool>("enable_bias");
    int embedding_size = ctx->Attrs().Get<int>("embedding_size");
    size_t weight_size = 4 * embedding_size * embedding_size;
    if (enable_bias) weight_size += 4 * embedding_size;
    auto weight_dims = ctx->GetInputDim("weight");
    size_t weight_tensor_size = 1;
    for (int i = 0; i < weight_dims.size(); ++i) {
      weight_tensor_size *= weight_dims[i];
    }

    PADDLE_ENFORCE_EQ(
        weight_tensor_size, weight_size,
        platform::errors::InvalidArgument(
            "The input tensor weight's size of MHAOp "
            "should be equal to 4*%d*%d (%d) + [4*%d when"
            " enable_bias==true, (total=%d)]."
            " But received weight's size = %d.",
            embedding_size, embedding_size, 4 * embedding_size * embedding_size,
            embedding_size, weight_size, weight_tensor_size));

    int batch_size = query_dims[0];
    auto qo_kv_seqlen_dims = ctx->GetInputDim("qo_kv_seqlen");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          qo_kv_seqlen_dims[0], 2 * batch_size,
          platform::errors::InvalidArgument(
              "The number of sequence length should be equal"
              " to 2*(batch size). The first batch elements are qo seqlen"
              " and the rest is kv seqlen"));
    }

    int seqlen = query_dims[1];
    if (ctx->HasInput("low_high_windows_host")) {
      auto low_high_windows = ctx->GetInputDim("low_high_windows_host");
      if (ctx->IsRuntime()) {
        PADDLE_ENFORCE_EQ(
            low_high_windows[0], 2 * seqlen,
            platform::errors::InvalidArgument(
                "The number of low_high_windows should be equal"
                " to 2*(sequence length). The first seqlen elements"
                " are low windows and the rest is high windows"));
      }
    }

    std::vector<int64_t> output_dims;
    for (int i = 0; i < query_dims.size(); ++i) {
      output_dims.push_back(query_dims[i]);
    }

    ctx->SetOutputDim("output", framework::make_ddim(output_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "query");
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
    AddInput("query", "");
    AddInput("key", "");
    AddInput("value", "");
    AddInput("weight", "");
    AddInput("qo_kv_seqlen", "");
    AddInput("qo_kv_seqlen_host", "").AsDispensable();
    AddInput("low_high_windows_host", "").AsDispensable();

    AddOutput("output", "");
    AddAttr<std::string>("cache_key", "");
    AddAttr<bool>("is_training", "").SetDefault(true);
    AddAttr<bool>("enable_bias", "").SetDefault(true);
    AddAttr<float>("pre_dropout_rate", "").SetDefault(0.0);
    AddAttr<float>("post_dropout_rate", "").SetDefault(0.0);
    AddAttr<int>("num_heads", "");
    AddAttr<float>("softmax_scaler", "");
    AddAttr<int>("embedding_size", "");
    AddAttr<int>("query_proj_size", "");
    AddAttr<int>("key_proj_size", "");
    AddAttr<int>("value_proj_size", "");
    AddAttr<int>("output_proj_size", "");
    AddAttr<int>("max_qo_seqlen", "");
    AddAttr<int>("max_kv_seqlen", "");
    // AddAttr<int>("beam_size", "Not supported currently.");

    AddComment(R"DOC(MHA OP)DOC");
  }
};

class MHAOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"query", /*->*/ "output"}};
    return m;
  }
};

class MHAGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("output")), "Input",
                   framework::GradVarName("output"), "mha");
    OP_INOUT_CHECK(ctx->HasInput("qo_kv_seqlen"), "Input", "qo_kv_seqlen",
                   "mha");

    std::string var_names[4] = {"query", "key", "value", "weight"};
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
      const framework::ExecutionContext &ctx) const {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "query");
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
    retv->SetInput("query", this->Input("query"));
    retv->SetInput("key", this->Input("key"));
    retv->SetInput("value", this->Input("value"));
    retv->SetInput("weight", this->Input("weight"));
    retv->SetInput("qo_kv_seqlen", this->Input("qo_kv_seqlen"));
    if (this->HasInput("low_high_windows_host")) {
      retv->SetInput("low_high_windows_host",
                     this->Input("low_high_windows_host"));
    }

    retv->SetInput(framework::GradVarName("output"),
                   this->OutputGrad("output"));
    retv->SetOutput(framework::GradVarName("query"), this->InputGrad("query"));
    retv->SetOutput(framework::GradVarName("key"), this->InputGrad("key"));
    retv->SetOutput(framework::GradVarName("value"), this->InputGrad("value"));
    retv->SetOutput(framework::GradVarName("weight"),
                    this->InputGrad("weight"));
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
    mha, ops::DummyMHAKernel<paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_CPU_KERNEL(
    mha_grad,
    ops::DummyMHAGradKernel<paddle::platform::CPUDeviceContext, float>);

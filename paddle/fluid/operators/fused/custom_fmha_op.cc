/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {
using Tensor = phi::DenseTensor;

class CustomFMHAOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // Check input
    OP_INOUT_CHECK(
        ctx->HasInput("QKV"), "Input", "QKV", "CustomFMHAOp");
    OP_INOUT_CHECK(
        ctx->HasInput("CuSeqLen"), "Input", "CuSeqLen", "CustomFMHAOp");
    OP_INOUT_CHECK(
        ctx->HasInput("HostSeqLen"), "Input", "HostSeqLen", "CustomFMHAOp");
    // Check output
    OP_INOUT_CHECK(
        ctx->HasOutput("CtxOut"), "Output", "CtxOut", "CustomFMHAOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("SOut"), "Output", "SOut", "CustomFMHAOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("DropoutMask"), "Output", "DropoutMask", "CustomFMHAOp");
    OP_INOUT_CHECK(
        ctx->HasOutput("DropoutOut"), "Output", "DropoutOut", "CustomFMHAOp");
    // Get dims of inputs
    auto x_shape = ctx->GetInputDim("QKV");
    auto y_shape = ctx->GetInputDim("CuSeqLen");
    auto host_y_shape = ctx->GetInputDim("HostSeqLen");
    // x_shape(fp16) = [total_tokens, 3, num_heads, head_size]
    // y_shape(int32) = [batch_size + 1]
    PADDLE_ENFORCE_EQ(
        x_shape.size(),
        4,
        platform::errors::InvalidArgument(
            "The Input tensor QKV's dimension of CustomFMHAOp "
            " should be 4, but got %d.",
            x_shape.size()));
    PADDLE_ENFORCE_EQ(
        y_shape.size(),
        1,
        platform::errors::InvalidArgument(
            "The Input tensor CuSeqLen's dimension of CustomFMHAOp "
            " should be 1, but got %d.",
            y_shape.size()));
    PADDLE_ENFORCE_EQ(
        host_y_shape.size(),
        1,
        platform::errors::InvalidArgument(
            "The Input tensor HostSeqLen's dimension of CustomFMHAOp "
            " should be 1, but got %d.",
            host_y_shape.size()));
    PADDLE_ENFORCE_EQ(
        x_shape[1],
        3,
        platform::errors::InvalidArgument(
            " The shape for input QKV should be [total_tokens, 3, num_heas, head_size]. "
            " but dims[1] got %d.",
            x_shape[1]));
//    PADDLE_ENFORCE_GT(
//        y_shape[0],
//        1,
//        platform::errors::InvalidArgument(
//            " The shape for CuSeqLen should be [batch_size + 1]. "
//            " but dims[0] got %d.",
//            y_shape[0]));
    // Get Attrs
    //    auto is_test = ctx->Attrs().Get<bool>("is_test");
    //    auto dropout_rate = ctx->Attrs().Get<float>("dropout_rate");
    //    auto zero_tensors = ctx->Attrs().Get<bool>("zero_tensors");
    //    auto use_fmha_mke_opt = ctx->Attrs().Get<bool>("use_fmha_mke_opt");
    //
    int total = x_shape[0];
    int num_heads = x_shape[2];
    int head_size = x_shape[3];
    int batch_size = y_shape[0] - 1;
    // Set dims of output
    std::vector<int64_t> ctx_out_shape = {total, num_heads, head_size};
    ctx->SetOutputDim("CtxOut", phi::make_ddim(ctx_out_shape));
    int max_seq_len = 512;
    std::vector<int64_t> s_out_shape = {
        batch_size, num_heads, max_seq_len, max_seq_len};
    ctx->SetOutputDim("SOut", phi::make_ddim(s_out_shape)); 
    ctx->SetOutputDim("DropoutMask", phi::make_ddim(s_out_shape));
    ctx->SetOutputDim("DropoutOut", phi::make_ddim(s_out_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto qkv_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "QKV");
    // qkv          -> float16
    // cu_seq_len   -> int
    // host_seq_len -> int
    // ctx_out      -> float16
    // s_out        -> float16
    // dropout_mask -> float16
    // dropout_out -> float16
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        qkv_data_type, ctx.GetPlace(), layout, library);
  }
};

class CustomFMHAOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("QKV", "float16[total_tokens, 3, num_heads, head_size]");
    AddInput("CuSeqLen", "int[batch_size + 1], lod on device");
    AddInput("HostSeqLen", "int[batch_size + 1], lod on host");
    AddOutput("CtxOut", "float16[total_tokens, num_heads, head_size]");
    AddOutput("SOut", "float16[batch_size, num_heads, max_seq_len, max_seq_len]");
    AddOutput("DropoutMask", "same shape as SOut");
    AddOutput("DropoutOut", "same shape as SOut");
    AddAttr<bool>("is_test", "is_test").SetDefault(false);
    AddAttr<float>("dropout_rate", "dropout_rate").SetDefault(0.0);
    AddAttr<bool>("zero_tensors", "zero_tensors").SetDefault(false);
    AddAttr<bool>("use_fmha_mke_opt", "use_fmha_mke_opt").SetDefault(false);
    AddComment(R"DOC(
FMHA: fused multi-head self-attention.
** This is only use for XPU, if has problems, concat liyupeng03@baidu.com **
)DOC");
  }
};

class CustomFMHAGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const {
    // Check input
    OP_INOUT_CHECK(
        ctx->HasInput("QKV"), "Input", "QKV", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("CuSeqLen"), "Input", "CuSeqLen", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("HostSeqLen"), "Input", "HostSeqLen", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("SOut"), "Input", "SOut", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("DropoutMask"), "Input", "DropoutMask", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("DropoutOut"), "Input", "DropoutOut", "CustomFMHAGradOp");
    OP_INOUT_CHECK(
        ctx->HasInput("DCtxOut"), "Input", "DCtxOut", "CustomFMHAGradOp");
    // Check output
    OP_INOUT_CHECK(
        ctx->HasOutput("DQKV"), "Output", "DQKV", "CustomFMHAGradOp");
    // Get dims of inputs
    auto qkv_shape = ctx->GetInputDim("QKV");
    // qkv_shape(fp16) = [total_tokens, 3, num_heads, head_size]
    // Set dims of output
    std::vector<int64_t> d_qkv_shape = {qkv_shape[0], qkv_shape[1], qkv_shape[2], qkv_shape[3]};
    ctx->SetOutputDim("DQKV", phi::make_ddim(d_qkv_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    auto qkv_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "QKV");
    // qkv          -> float16
    // cu_seq_len   -> int
    // host_seq_len -> int
    // ctx_out      -> float16
    // s_out        -> float16
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    return framework::OpKernelType(
        qkv_data_type, ctx.GetPlace(), layout, library);
  }
};

class CustomFMHAGradOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("QKV", "float16[total_tokens, 3, num_heads, head_size]");
    AddInput("CuSeqLen", "int[batch_size + 1], lod on device");
    AddInput("HostSeqLen", "int[batch_size + 1], lod on host");
    AddInput("SOut", "float16[batch_size, num_heads, max_seq_len, max_seq_len]");
    AddInput("DropoutMask", "same shape as SOut");
    AddInput("DropoutOut", "same shape as SOut");
    AddInput("DCtxOut", "float16[total_tokens, num_heads, head_size]");
    AddOutput("DQKV", "float16[total_tokens, 3, num_heads, head_size]");
    AddAttr<bool>("is_test", "is_test").SetDefault(false);
    AddAttr<float>("dropout_rate", "dropout_rate").SetDefault(0.0);
    AddAttr<bool>("zero_tensors", "zero_tensors").SetDefault(false);
    AddAttr<bool>("use_fmha_mke_opt", "use_fmha_mke_opt").SetDefault(false);
    AddComment(R"DOC(
Grad Op of FMHA: fused multi-head self-attention.
** This is only use for XPU, if has problems, concat liyupeng03@baidu.com **
)DOC");
  }
};

template <typename T>
class CustomFMHAOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("custom_fmha_grad");
    op->SetInput("QKV", this->Input("QKV"));
    op->SetInput("CuSeqLen", this->Input("CuSeqLen"));
    op->SetInput("HostSeqLen", this->Input("HostSeqLen"));
    op->SetInput("SOut", this->Output("SOut"));
    op->SetInput("DropoutMask", this->Output("DropoutMask"));
    op->SetInput("DropoutOut", this->Output("DropoutOut"));
    op->SetInput("DCtxOut", this->OutputGrad("CtxOut"));
    op->SetOutput("DQKV", this->InputGrad("QKV"));
    
    op->SetAttrMap(this->Attrs());
  }
};

class CustomFMHAOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"QKV", /*->*/ "CtxOut"}, {"QKV", /*->*/ "SOut"}};
    return m;
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(custom_fmha,
                  ops::CustomFMHAOp,
                  ops::CustomFMHAOpMaker,
                  ops::CustomFMHAOpInferVarType,
                  ops::CustomFMHAOpGradMaker<paddle::framework::OpDesc>,
                  ops::CustomFMHAOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(custom_fmha_grad,
                  ops::CustomFMHAGradOp,
                  ops::CustomFMHAGradOpMaker);

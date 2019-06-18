/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cmath>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

class SearchAttentionPaddingMaskOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("Y",
             "Y (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddAttr<int>("pad_id", "pad_id").SetDefault(0).EqualGreaterThan(0);
	AddAttr<float>("mask", "mask").SetDefault(0.0);

    AddOutput("Out",
              "Out (LoDTensor, default LoDTensor<float>) Output variable");
    AddOutput(
        "pad_begin",
        "pad_begin (LoDTensor, default LoDTensor<float>) Output variable");

    AddComment(R"DOC(
  SearchAttentionPaddingMask

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchAttentionPaddingMaskOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("pad_begin"),
                   "pad_begin(Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

    auto y_dims = ctx->GetInputDim("Y");
    PADDLE_ENFORCE_EQ(y_dims.size(), 2, "Y should be 2-D tensor");

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      const auto& x_lod = x_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!x_lod.empty(), "The Input(X) must hold lod info.");
      const auto& x_lod_0 = x_lod[0];
      PADDLE_ENFORCE_GE(x_lod_0.size(), 2,
                        "The Input(X)'s lod info is corrupted.");
      PADDLE_ENFORCE_EQ(
          x_dims[0], static_cast<int64_t>(x_lod_0.back()),
          "The Input(X)'s lod info mismatches the actual tensor shape.");

      framework::Variable* y_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Y")[0]);
      const auto& y_lod = y_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!y_lod.empty(), "The Input(Y) must hold lod info.");
      const auto& y_lod_0 = y_lod[0];
      PADDLE_ENFORCE_GE(y_lod_0.size(), 2,
                        "The Input(Y)'s lod info is corrupted.");
      PADDLE_ENFORCE_EQ(
          y_dims[0], static_cast<int64_t>(y_lod_0.back()),
          "The Input(Y)'s lod info mismatches the actual tensor shape.");
    } else {
      // compile time
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1, x_dims[1]}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class CPUSearchAttentionPaddingMaskOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* _pad_begin = ctx.Output<LoDTensor>("pad_begin");

    int _pad_id = ctx.Attr<int>("pad_id");
    float _mask = ctx.Attr<float>("mask");

    const auto src_len = static_cast<int64_t>(bottom1->lod()[0][1]);
    PADDLE_ENFORCE_EQ(src_len, bottom0->dims()[1],
                      "Mismatch source length, expect: %d get: %d", src_len,
                      bottom0->dims()[1]);
    const int att_batch = bottom0->lod()[0].size() - 1;
    const int src_batch = bottom1->lod()[0].size() - 1;
    PADDLE_ENFORCE_EQ(att_batch % src_batch, 0,
                      "Mismatch batch size, bottom0: %d, bottom1: %d",
                      att_batch, src_batch);

    _pad_begin->Resize(framework::make_ddim({src_batch}));
    int* pad_begin = _pad_begin->mutable_data<int>(ctx.GetPlace());
    for (int i = 0; i < src_batch; ++i) {
      // bottom data is padded to be aligned
      const auto* src_data = bottom1->data<T>() + src_len * i;
      int index = src_len - 1;
      for (; index >= 0 && _pad_id == static_cast<int>(src_data[index]);
           --index) {
      }
      pad_begin[i] = index + 1;
    }

    top->Resize(bottom0->dims());
    const auto att_len = static_cast<int64_t>(bottom0->lod()[0][1]);
    auto* top_data = top->mutable_data<T>(ctx.GetPlace());
    memcpy(top_data, bottom0->data<T>(),
           bottom0->dims()[0] * bottom0->dims()[1] * sizeof(T));
    for (int i = 0; i < att_batch; ++i) {
      for (int j = 0; j < att_len; ++j) {
        top_data =
            top->mutable_data<T>(ctx.GetPlace()) + src_len * (att_len * i + j);
        int src_idx = i % src_batch;
        for (int k = pad_begin[src_idx]; k < src_len; ++k) {
          top_data[k] = _mask;
        }
      }
    }
  }
};

class SearchAttentionPaddingMaskGradOpMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType("search_attention_padding_mask_grad");
    op_desc_ptr->SetInput("X", Input("X"));
    op_desc_ptr->SetInput("Y", Input("Y"));
    op_desc_ptr->SetInput("pad_begin", Output("pad_begin"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op_desc_ptr->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

class SearchAttentionPaddingMaskOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("pad_begin"),
                   "Input(pad_begin) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("X"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class CPUSearchAttentionPaddingMaskOPGradKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* _pad_begin = ctx.Input<LoDTensor>("pad_begin");
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    const int* pad_begin = _pad_begin->data<int>();
    const auto att_batch = bottom0->lod()[0].size() - 1;
    const auto src_batch = bottom1->lod()[0].size() - 1;

    const auto att_len = bottom0->lod()[0][1];
    const auto src_len = bottom1->lod()[0][1];

    auto* att_diff = d_x->mutable_data<T>(ctx.GetPlace());
    memcpy(att_diff, d_out->data<T>(),
           d_out->dims()[0] * d_out->dims()[1] * sizeof(T));
    for (int i = 0; i < att_batch; ++i) {
      for (int j = 0; j < att_len; ++j) {
        int src_idx = i % src_batch;
        att_diff = d_x->mutable_data<T>(ctx.GetPlace()) +
                   src_len * (att_len * i + j) + pad_begin[src_idx];
        memset(att_diff, 0, (src_len - pad_begin[src_idx]) * sizeof(T));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_attention_padding_mask,
                  ops::SearchAttentionPaddingMaskOP,
                  ops::SearchAttentionPaddingMaskOpMaker,
                  ops::SearchAttentionPaddingMaskGradOpMaker);
REGISTER_OPERATOR(search_attention_padding_mask_grad,
                  ops::SearchAttentionPaddingMaskOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_attention_padding_mask,
    ops::CPUSearchAttentionPaddingMaskOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchAttentionPaddingMaskOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_attention_padding_mask_grad,
    ops::CPUSearchAttentionPaddingMaskOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchAttentionPaddingMaskOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

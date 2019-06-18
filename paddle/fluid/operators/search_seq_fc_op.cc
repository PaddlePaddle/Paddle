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
#include "search_compute.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

class SearchSeqFCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("W", "W (Tensor)");
    AddInput("b", "b (LoDTensor)");
    AddAttr<int>("out_size", "out_size: the output size")
        .SetDefault(0)
        .EqualGreaterThan(1);
    AddAttr<bool>("has_bias", "true or false").SetDefault(true);

    AddOutput("Out",
              "Out (LoDTensor, default LoDTensor<float>) Output variable");

    AddComment(R"DOC(
  SearchSeqFC

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchSeqFCOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "W(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
    int out_size = ctx->Attrs().Get<int>("out_size");
    bool has_bias = ctx->Attrs().Get<bool>("has_bias");


    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2, "W should be 2-D tensor");

    PADDLE_ENFORCE_EQ(w_dims[0], out_size,
                      "wrong shape: w_dims[0] != out_size");

    PADDLE_ENFORCE_EQ(w_dims[1], x_dims[1],
                      "wrong shape: w_dims[1] != x_dims[1]");

    if (has_bias) {
      PADDLE_ENFORCE(ctx->HasInput("b"), "b(Input) should not be null.");
      auto b_dims = ctx->GetInputDim("b");
      PADDLE_ENFORCE_EQ(b_dims.size(), 1, "b should be 1-D tensor");
    }

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
    } else {
      // compile time
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1, out_size}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqFCOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    auto* b = ctx.Input<Tensor>("b");
    auto* top = ctx.Output<LoDTensor>("Out");
    bool _bias_term = ctx.Attr<bool>("has_bias");

    int _out = w->dims()[0];
    int _in = w->dims()[1];
    int res_num = bottom->dims()[0];

    top->Resize(framework::make_ddim({res_num, _out}));
    const auto* bottom_data = bottom->data<T>();
    auto* top_data = top->mutable_data<T>(ctx.GetPlace());
    const auto* weights = w->data<T>();

    call_gemm(ctx, CblasNoTrans, CblasTrans, res_num, _out, _in, (T)1.0,
              bottom_data, weights, (T)0.0, top_data);

    if (_bias_term) {
      const auto* bias = b->data<T>();;
      for (int i = 0; i < res_num; ++i) {
        sse_eltadd(top_data + i * _out, bias, top_data + i * _out, _out);
      }
    }
  }
};

class SearchSeqFCOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("b"), "Input(b) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) of SequencePadGradOp should not be null.");

    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")));
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("W")));

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));

    bool has_bias = ctx->Attrs().Get<bool>("has_bias");
    if (has_bias) {
      PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("b")));
      ctx->SetOutputDim(framework::GradVarName("b"), ctx->GetInputDim("b"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqFCOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    bool _bias_term = ctx.Attr<bool>("has_bias");

    int _out = w->dims()[0];
    int _in = w->dims()[1];
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_w = ctx.Output<Tensor>(framework::GradVarName("W"));

    int res_num = bottom->dims()[0];

    const auto* top_diff = d_out->data<T>();
    const auto* bottom_data = bottom->data<T>();
    auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());
    const auto* weights = w->data<T>();
    auto* weights_diff = d_w->mutable_data<T>(ctx.GetPlace());

    call_gemm(ctx, CblasTrans, CblasNoTrans, _out, _in, res_num, (T)1.0,
              top_diff, bottom_data, (T)0.0, weights_diff);
    call_gemm(ctx, CblasNoTrans, CblasNoTrans, res_num, _in, _out, (T)1.0,
              top_diff, weights, (T)0.0, bottom_diff);

    if (_bias_term) {
      auto* d_b = ctx.Output<Tensor>(framework::GradVarName("b"));
      auto* bias_diff = d_b->mutable_data<T>(ctx.GetPlace());
      memset(bias_diff, (T)0.0, _out * sizeof(T));
      for (int i = 0; i < res_num; ++i) {
        sse_eltadd(bias_diff, top_diff + i * _out, bias_diff, _out);
      }
    }
        
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_seq_fc, ops::SearchSeqFCOP, ops::SearchSeqFCOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_seq_fc_grad, ops::SearchSeqFCOpGrad);

REGISTER_OP_CPU_KERNEL(search_seq_fc,
                       ops::CPUSearchSeqFCOPKernel<plt::CPUDeviceContext, float>
                       //     ops::CPUSearchSeqFCOPKernel<plt::CPUDeviceContext,
                       //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_seq_fc_grad,
    ops::CPUSearchSeqFCOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqFCOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

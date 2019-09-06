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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

class SearchSeqDepaddingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Pad",
             "Pad (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("Src",
             "Src (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddOutput("Out", "Out");

    AddComment(R"DOC(
  SearchSeqDepadding

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchSeqDepaddingOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Pad"), "Pad(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Src"), "Src(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");

    auto pad_dims = ctx->GetInputDim("Pad");
    PADDLE_ENFORCE_EQ(pad_dims.size(), 2,
                      "The rank of Pad(Input) should be 2.");

    auto src_dims = ctx->GetInputDim("Src");
    PADDLE_ENFORCE_EQ(src_dims.size(), 2,
                      "The rank of Src(Input) should be 2.");

    if (ctx->IsRuntime()) {
      framework::Variable* pad_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Pad")[0]);
      const auto& pad_lod = pad_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!pad_lod.empty(), "The Input(Pad) must hold lod info.");
      const auto& pad_lod_0 = pad_lod[0];
      PADDLE_ENFORCE_GE(pad_lod_0.size(), 2,
                        "The Input(Pad)'s lod info is corrupted.");
      PADDLE_ENFORCE_EQ(
          pad_dims[0], static_cast<int64_t>(pad_lod_0.back()),
          "The Input(Pad)'s lod info mismatches the actual tensor shape.");

      framework::Variable* src_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Src")[0]);
      const auto& src_lod = src_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!src_lod.empty(), "The Input(Src) must hold lod info.");
      const auto& src_lod_0 = src_lod[0];
      PADDLE_ENFORCE_GE(src_lod_0.size(), 2,
                        "The Input(Src)'s lod info is corrupted.");
      PADDLE_ENFORCE_EQ(
          src_dims[0], static_cast<int64_t>(src_lod_0.back()),
          "The Input(Src)'s lod info mismatches the actual tensor shape.");
    } else {
      // compile time
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1, pad_dims[1]}));
    //ctx->ShareLoD("Src", /*->*/ "Out");
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqDepaddingOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("Pad");
    auto* bottom1 = ctx.Input<LoDTensor>("Src");
    auto* top0 = ctx.Output<LoDTensor>("Out");

    const int pad_batch = bottom0->lod()[0].size() - 1;
    const int src_batch = bottom1->lod()[0].size() - 1;
    PADDLE_ENFORCE_EQ(pad_batch % src_batch, 0,
                      "Mismatch batch size, bottom0: %d, bottom1: %d",
                      pad_batch, src_batch);

    const auto& src_offset = bottom1->lod()[0];
    const auto& pad_offset = bottom0->lod()[0];
    const int src_cap_l = bottom1->dims()[0];
    const int pad_cap_e = bottom0->dims()[1];

    framework::LoD top0_lod;
    top0_lod.push_back(src_offset);
    top0->set_lod(top0_lod);
    top0->Resize(framework::make_ddim({src_cap_l, pad_cap_e}));

    const auto* bottom_data = bottom0->data<T>();
    auto* top_data = top0->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < src_batch; ++i) {
      const int src_i_l = src_offset[i + 1] - src_offset[i];
      const int pad_i_l = pad_offset[i + 1] - pad_offset[i];
      PADDLE_ENFORCE_GE(
          pad_i_l, src_i_l,
          "the length of padding seq input is less than source seq input.");
      memcpy(top_data + src_offset[i] * pad_cap_e,
             bottom_data + pad_offset[i] * pad_cap_e,
             src_i_l * pad_cap_e * sizeof(T));
    }
  }
};

class SearchSeqDepaddingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Pad"), "Input(Pad) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Src"), "Input(Src) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    if (ctx->HasOutput(framework::GradVarName("Pad"))) {
      ctx->SetOutputDim(framework::GradVarName("Pad"), ctx->GetInputDim("Pad"));
      ctx->ShareLoD("Pad", /*->*/ framework::GradVarName("Pad"));
    }
    if (ctx->HasOutput(framework::GradVarName("Src"))) {
      ctx->SetOutputDim(framework::GradVarName("Src"), ctx->GetInputDim("Src"));
      ctx->ShareLoD("Src", /*->*/ framework::GradVarName("Src"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqDepaddingOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("Pad");
    auto* bottom1 = ctx.Input<LoDTensor>("Src");
    auto* d_bottom0 = ctx.Output<LoDTensor>(framework::GradVarName("Pad"));
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));

    const int src_batch = bottom1->lod()[0].size() - 1;
    const auto& src_offset = bottom1->lod()[0];
    const auto& pad_offset = bottom0->lod()[0];
    const int pad_cap_e = bottom0->dims()[1];

    const auto* top_diff = d_out->data<T>();
    auto* bottom_diff = d_bottom0->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < src_batch; i++) {
      const int src_i_l = src_offset[i + 1] - src_offset[i];
      const int pad_i_l = pad_offset[i + 1] - pad_offset[i];
      PADDLE_ENFORCE_GE(
          pad_i_l, src_i_l,
          "the length of padding seq input is less than source seq input.");

      memcpy(bottom_diff + pad_offset[i] * pad_cap_e,
             top_diff + src_offset[i] * pad_cap_e,
             src_i_l * pad_cap_e * sizeof(T));
      memset(bottom_diff + (pad_offset[i] + src_i_l) * pad_cap_e, 0,
             (pad_i_l - src_i_l) * pad_cap_e * sizeof(T));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_seq_depadding, ops::SearchSeqDepaddingOP,
                  ops::SearchSeqDepaddingOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_seq_depadding_grad, ops::SearchSeqDepaddingOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_seq_depadding,
    ops::CPUSearchSeqDepaddingOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqDepaddingOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_seq_depadding_grad,
    ops::CPUSearchSeqDepaddingOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqDepaddingOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

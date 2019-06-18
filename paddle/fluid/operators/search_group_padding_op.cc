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

class SearchGroupPaddingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddAttr<int>("pad_id", "pad_id").SetDefault(0).EqualGreaterThan(0);

    AddOutput("Out_emb_padding", "Out_emb_padding");
    AddOutput("Out_new", "Out_new");
    AddOutput("Out_padding", "Out_padding");

    AddComment(R"DOC(
  SearchGroupPadding

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchGroupPaddingOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out_emb_padding"),
                   "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out_new"),
                   "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out_padding"),
                   "Out(Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

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
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_GE(x_desc->GetLoDLevel(), 1);
    }

    ctx->SetOutputDim("Out_emb_padding", framework::make_ddim({-1, x_dims[1]}));
    ctx->SetOutputDim("Out_new", framework::make_ddim({x_dims[0], 1}));
    // ctx->ShareLoD("X", /*->*/ "Out_new");
    ctx->SetOutputDim("Out_padding", framework::make_ddim({-1, 1}));
  }
};

template <typename DeviceContext, typename T>
class CPUSearchGroupPaddingOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* top0 = ctx.Output<LoDTensor>("Out_emb_padding");
    auto* top1 = ctx.Output<LoDTensor>("Out_new");
    auto* top2 = ctx.Output<LoDTensor>("Out_padding");

    int _pad_id = ctx.Attr<int>("pad_id");

    int batch = bottom0->lod()[0].size() - 1;
    int dim0 = bottom0->dims()[0];
    int dim1 = bottom0->dims()[1];  // dim1 is usually the embedding size

    const auto offset = bottom0->lod()[0];
    int max_seq = 0;
    for (int i = 0; i < batch; ++i) {
      if (offset[i + 1] - offset[i] > max_seq) {
        max_seq = offset[i + 1] - offset[i];
      }
    }

    std::vector<size_t> new_offset;
    new_offset.resize(batch + 1);

    for (int i = 0; i < batch + 1; ++i) {
      new_offset[i] = i * max_seq;
    }

    // for padding data
    framework::LoD top0_lod;
    top0_lod.push_back(new_offset);
    top0->set_lod(top0_lod);
    top0->Resize(framework::make_ddim({batch * max_seq, dim1}));

    // for origin input id
    // already set by ShareLoD in InferShape
          framework::LoD top1_lod;
          top1_lod.push_back(offset);
          top1->set_lod(top1_lod);
    top1->Resize(framework::make_ddim({dim0, 1}));
    memset(top1->mutable_data<T>(ctx.GetPlace()), 0,
           top1->dims()[0] * top1->dims()[1] * sizeof(T));

    // for padding input id
    framework::LoD top2_lod;
    top2_lod.push_back(new_offset);
    top2->set_lod(top2_lod);
    top2->Resize(framework::make_ddim({batch * max_seq, 1}));

    // copy data
    const auto* bottom_data = bottom0->data<T>();
    auto* top_data = top0->mutable_data<T>(ctx.GetPlace());
    auto* top_padding_input_data = top2->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < batch; i++) {
      const int copy_step = offset[i + 1] - offset[i];
      const int start = i * max_seq;
      memcpy(top_data + start * dim1, bottom_data + offset[i] * dim1,
             copy_step * dim1 * sizeof(T));
      memset(top_data + (start + copy_step) * dim1, 0,
             (max_seq - copy_step) * dim1 * sizeof(T));
      // for padding input id
      memset(top_padding_input_data + start, 0, copy_step * sizeof(T));
      for (int j = start + copy_step; j < start + max_seq; j++) {
        top_padding_input_data[j] = static_cast<T>(_pad_id);
      }
    }
  }
};

class SearchGroupPaddingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput(framework::GradVarName("Out_emb_padding")),
        "Input(Out_emb_padding@GRAD) of SequencePadGradOp should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchGroupPaddingOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* top0 = ctx.Input<LoDTensor>("Out_emb_padding");
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_out =
        ctx.Input<LoDTensor>(framework::GradVarName("Out_emb_padding"));

    int batch = bottom0->lod()[0].size() - 1;
    int dim1 = bottom0->dims()[1];  // dim1 is usually the embedding size

    auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());
    const auto* top_diff = d_out->data<T>();
    const auto offset = bottom0->lod()[0];
    const auto top_offset = top0->lod()[0];
    for (int i = 0; i < batch; i++) {
      const int step = offset[i + 1] - offset[i];
      memcpy(bottom_diff + offset[i] * dim1, top_diff + top_offset[i] * dim1,
             step * dim1 * sizeof(T));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_group_padding, ops::SearchGroupPaddingOP,
                  ops::SearchGroupPaddingOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_group_padding_grad, ops::SearchGroupPaddingOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_group_padding,
    ops::CPUSearchGroupPaddingOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchGroupPaddingOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_group_padding_grad,
    ops::CPUSearchGroupPaddingOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchGroupPaddingOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

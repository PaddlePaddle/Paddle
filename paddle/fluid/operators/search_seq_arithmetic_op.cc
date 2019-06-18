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

class SearchSeqArithmeticOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("Y",
             "Y (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddAttr<int>("op_type", "operation type: 1: add; 2: sub; 3: mul")
        .SetDefault(0)
        .EqualGreaterThan(1);

    AddOutput("Out",
              "Out (LoDTensor, default LoDTensor<float>) Output variable");

    AddComment(R"DOC(
  SearchSeqArithmetic

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchSeqArithmeticOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");

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

      PADDLE_ENFORCE_EQ(x_lod_0.size(), y_lod_0.size(),
                        "The Length of X and Y must be equal.");
    } else {
      // compile time
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_GE(x_desc->GetLoDLevel(), 1);
      framework::VarDesc* y_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_GE(y_desc->GetLoDLevel(), 1);
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1, x_dims[1]}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqArithmeticOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* top = ctx.Output<LoDTensor>("Out");

    int _op_type = ctx.Attr<int>("op_type");

    auto len1 = bottom0->dims()[0] * bottom0->dims()[1];
    auto len2 = bottom1->dims()[0] * bottom1->dims()[1];
    const auto* bottom_data0 = bottom0->data<T>();
    const auto* bottom_data1 = bottom1->data<T>();
    // already set by ShareLoD in InferShape
    //        framework::LoD top_lod;
    //        top_lod.push_back(offset);
    //        top->set_lod(top_lod);
    top->Resize(framework::make_ddim({bottom0->dims()[0], bottom0->dims()[1]}));
    auto* top_data = top->mutable_data<T>(ctx.GetPlace());

    switch (_op_type) {
      case 1:  // addition: top[0] = bottom[0] + bottom[1]
        if (len1 > len2) {
          sse_eltadd(bottom_data0, bottom_data1, top_data, len2);
          memcpy(&top_data[len2], &bottom_data0[len2],
                 (len1 - len2) * sizeof(T));
        } else {
          sse_eltadd(bottom_data0, bottom_data1, top_data, len1);
        }
        break;
      case 2:  // substraction: top[0] = bottom[0] - bottom[1]
        memcpy(top_data, bottom_data0, len1 * sizeof(T));
        if (len1 > len2) {
          sse_axpy(bottom_data1, top_data, len2, (T)-1.0);
        } else {
          sse_axpy(bottom_data1, top_data, len1, (T)-1.0);
        }
        break;
      case 3:  // multiplication: top[0] = bottom[0] * bottom[1]
        if (len1 > len2) {
          sse_eltmul(bottom_data0, bottom_data1, top_data, len2);
          memcpy(&top_data[len2], &bottom_data0[len2],
                 (len1 - len2) * sizeof(T));
        } else {
          sse_eltmul(bottom_data0, bottom_data1, top_data, len1);
        }
        break;
    }
  }
};

class SearchSeqArithmeticOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
      ctx->ShareLoD("Y", /*->*/ framework::GradVarName("Y"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqArithmeticOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Output<LoDTensor>(framework::GradVarName("Y"));
    int _op_type = ctx.Attr<int>("op_type");

    auto len1 = bottom0->dims()[0] * bottom0->dims()[1];
    auto len2 = bottom1->dims()[0] * bottom1->dims()[1];
    auto* bottom_diff0 = d_x->mutable_data<T>(ctx.GetPlace());
    auto* bottom_diff1 = d_y->mutable_data<T>(ctx.GetPlace());
    const auto* top_diff = d_out->data<T>();
    const auto* bottom_data0 = bottom0->data<T>();
    const auto* bottom_data1 = bottom1->data<T>();

    switch (_op_type) {
      case 1:  // addition
        memcpy(bottom_diff0, top_diff, len1 * sizeof(T));
        if (len1 >= len2) {
          memcpy(bottom_diff1, top_diff, len2 * sizeof(T));
        } else {
          memset(bottom_diff1, 0, len2 * sizeof(T));
          memcpy(bottom_diff1, top_diff, len1 * sizeof(T));
        }
        break;
      case 2:  // substraction
        memcpy(bottom_diff0, top_diff, len1 * sizeof(T));
        if (len1 >= len2) {
          sse_axpy_noadd(top_diff, bottom_diff1, len2, (T)-1.0);
        } else {
          memset(bottom_diff1, 0, len2 * sizeof(T));
          sse_axpy_noadd(top_diff, bottom_diff1, len1, (T)-1.0);
        }
        break;
      case 3:  // multiplication
        if (len1 >= len2) {
          memcpy(bottom_diff0, top_diff, len1 * sizeof(T));
          sse_eltmul(top_diff, bottom_data1, bottom_diff0, len2);
          sse_eltmul(top_diff, bottom_data0, bottom_diff1, len2);
        } else {
          sse_eltmul(top_diff, bottom_data1, bottom_diff0, len1);
          memset(bottom_diff1, 0, len2 * sizeof(T));
          sse_eltmul(top_diff, bottom_data0, bottom_diff1, len1);
        }
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_seq_arithmetic, ops::SearchSeqArithmeticOP,
                  ops::SearchSeqArithmeticOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_seq_arithmetic_grad, ops::SearchSeqArithmeticOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_seq_arithmetic,
    ops::CPUSearchSeqArithmeticOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqArithmeticOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_seq_arithmetic_grad,
    ops::CPUSearchSeqArithmeticOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqArithmeticOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

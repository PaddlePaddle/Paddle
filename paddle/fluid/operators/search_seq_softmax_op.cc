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

class SearchSeqSoftmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddAttr<int>("alg", "operation type: 0: accurate; 1: log; others: invalid")
        .SetDefault(0)
        .EqualGreaterThan(0);

    AddOutput("Out",
              "Out (LoDTensor, default LoDTensor<float>) Output variable");
    AddOutput("Out_log",
              "Out_log (LoDTensor, default LoDTensor<float>) Output variable");

    AddComment(R"DOC(
  SearchSeqSoftmax

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchSeqSoftmaxOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out_log"), "Out_log(Output) should not be null.");

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
    }

    ctx->SetOutputDim("Out", framework::make_ddim({-1, x_dims[1]}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqSoftmaxOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* top0 = ctx.Output<LoDTensor>("Out");
    auto* _prob = ctx.Output<LoDTensor>("Out_log");
    int _output_log = ctx.Attr<int>("alg");

    int seq_size = bottom0->dims()[0];
    int dim = bottom0->dims()[1];
    const auto offset_vec = bottom0->lod()[0];
    top0->Resize(framework::make_ddim({seq_size, dim}));
    const auto* bottom_data = bottom0->data<T>();
    auto* top_data = top0->mutable_data<T>(ctx.GetPlace());

    for (int i = 0; i < seq_size; ++i) {
      int offset = i * dim;
      auto max_val =
          *std::max_element(bottom_data + offset, bottom_data + offset + dim);
      max_val *= -1;
      sse_add_scalar(bottom_data + offset, top_data + offset, dim, max_val);
      for (int j = 0; j < dim; ++j) {
        top_data[offset + j] = std::exp(top_data[offset + j]);
      }
      T sum;
      sse_sum(top_data + offset, sum, dim);
      sum = 1.0 / sum;
      sse_scale(top_data + offset, top_data + offset, dim, sum);
    }

    if (_output_log) {
      const int size = top0->dims()[0] * top0->dims()[1];
      _prob->Resize(framework::make_ddim({size}));
      auto* prob_data = _prob->mutable_data<T>(ctx.GetPlace());
      memcpy(prob_data, top_data, size * sizeof(T));
      for (int i = 0; i < size; ++i) {
        top_data[i] = std::log(std::max(prob_data[i], X_MIN));
      }
    } else {
      _prob->Resize(framework::make_ddim({1}));
      _prob->mutable_data<T>(ctx.GetPlace());
    }
  }
};

class SearchSeqSoftmaxOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchSeqSoftmaxOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* top0 = ctx.Input<LoDTensor>("Out");
    auto* _prob = ctx.Input<LoDTensor>("Out_log");
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    int _output_log = ctx.Attr<int>("alg");

    int seq_size = bottom0->dims()[0];
    int dim = bottom0->dims()[1];
    const auto* top_diff = d_out->data<T>();
    const auto* top_data = top0->data<T>();
    auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());

    if (_output_log) {
      const auto* prob_data = _prob->data<T>();
      Tensor buffer_diff;
      buffer_diff.Resize(_prob->dims());
      auto* prob_diff = buffer_diff.mutable_data<T>(ctx.GetPlace());

      const int size = top0->dims()[0] * top0->dims()[1];
      PADDLE_ENFORCE_EQ(size, _prob->dims()[0] * _prob->dims()[1], "top_size should be eq to _prob_size");
      for (int i = 0; i < size; ++i) {
        prob_diff[i] = top_diff[i] / std::max(prob_data[i], X_MIN);
      }
      top_diff = prob_diff;
      top_data = prob_data;
    }

    for (int i = 0; i < seq_size; ++i) {
      int offset = i * dim;
      T ip_d_t;
      sse_ip(top_diff + offset, top_data + offset, dim, ip_d_t);
      ip_d_t *= -1;
      sse_add_scalar(top_diff + offset, bottom_diff + offset, dim, ip_d_t);
      sse_eltmul(top_data + offset, bottom_diff + offset, bottom_diff + offset,
                 dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_seq_softmax, ops::SearchSeqSoftmaxOP,
                  ops::SearchSeqSoftmaxOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_seq_softmax_grad, ops::SearchSeqSoftmaxOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_seq_softmax,
    ops::CPUSearchSeqSoftmaxOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqSoftmaxOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_seq_softmax_grad,
    ops::CPUSearchSeqSoftmaxOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchSeqSoftmaxOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

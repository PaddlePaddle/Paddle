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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "paddle/fluid/operators/match_matrix_tensor_op.h"
#include "paddle/fluid/operators/search_compute.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

void MatchMatrixTensorOP::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                    "X(Input) of MatchMatrix should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                    "Y(Input) of MatchMatrix should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                    "W(Input) of MatchMatrix should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                    "Out(Output) of MatchMatrix should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasOutput("Tmp"), true,
                    "Tmp(Output) of MatchMatrix should not be null.");

  auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                    "The rank of Input(X) can't be less than 2.");

  auto y_dims = ctx->GetInputDim("Y");
  PADDLE_ENFORCE_EQ(y_dims.size(), 2,
                    "The rank of Input(Y) can't be less than 2.");

  auto w_dims = ctx->GetInputDim("W");
  PADDLE_ENFORCE_EQ(w_dims.size(), 3UL, "W should be 3-D tensor");

  int dim_t = ctx->Attrs().Get<int>("dim_t");
  PADDLE_ENFORCE_EQ(w_dims[0], x_dims[1],
                    "W 's shape must satisfy: W[0] = X[1]");
  PADDLE_ENFORCE_EQ(w_dims[1], dim_t, "W 's shape must satisfy: W[1] = dim_t");
  PADDLE_ENFORCE_EQ(w_dims[2], y_dims[1],
                    "W 's shape must satisfy: W[2] = Y[1]");

  int64_t out_dim_0 = -1;
  int64_t tmp_dim_0 = -1;
  if (ctx->IsRuntime()) {
    framework::Variable* x_var =
        boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
    const auto& x_lod = x_var->Get<LoDTensor>().lod();
    PADDLE_ENFORCE_EQ(x_lod.empty(), false, "The Input(X) must hold lod info.");
    const auto& x_lod_0 = x_lod[0];
    PADDLE_ENFORCE_GE(x_lod_0.size(), 2,
                      "The Input(X)'s lod info is corrupted.");
    PADDLE_ENFORCE_EQ(
        x_dims[0], static_cast<int64_t>(x_lod_0.back()),
        "The Input(X)'s lod info mismatches the actual tensor shape.");

    framework::Variable* y_var =
        boost::get<framework::Variable*>(ctx->GetInputVarPtrs("Y")[0]);
    const auto& y_lod = y_var->Get<LoDTensor>().lod();
    PADDLE_ENFORCE_EQ(y_lod.empty(), false, "The Input(Y) must hold lod info.");
    const auto& y_lod_0 = y_lod[0];
    PADDLE_ENFORCE_GE(y_lod_0.size(), 2,
                      "The Input(Y)'s lod info is corrupted.");
    PADDLE_ENFORCE_EQ(
        y_dims[0], static_cast<int64_t>(y_lod_0.back()),
        "The Input(Y)'s lod info mismatches the actual tensor shape.");

    PADDLE_ENFORCE_EQ(x_lod_0.size(), y_lod_0.size(),
                      "The Length of X and Y must be equal.");

    out_dim_0 = 0;
    for (size_t i = 1; i < x_lod_0.size(); i++) {
      int64_t x_len = x_lod_0[i] - x_lod_0[i - 1];
      int64_t y_len = y_lod_0[i] - y_lod_0[i - 1];
      out_dim_0 += (x_len * y_len);
    }
    out_dim_0 *= dim_t;

    tmp_dim_0 = x_dims[0] * dim_t * x_dims[1];
  } else {
    // compile time
    framework::VarDesc* x_desc =
        boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
    PADDLE_ENFORCE_GE(x_desc->GetLoDLevel(), 1);
    framework::VarDesc* y_desc =
        boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("Y")[0]);
    PADDLE_ENFORCE_GE(y_desc->GetLoDLevel(), 1);
    ctx->ShareLoD("X", "Out");
  }

  std::vector<int64_t> out_dims_vec{out_dim_0};
  out_dims_vec.push_back(1);
  std::vector<int64_t> tmp_dims_vec{tmp_dim_0};
  tmp_dims_vec.push_back(1);
  ctx->SetOutputDim("Out", framework::make_ddim(out_dims_vec));
  ctx->SetOutputDim("Tmp", framework::make_ddim(tmp_dims_vec));
}

void MatchMatrixTensorOpGrad::InferShape(
    framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                    "Input(X) of SequencePadGradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("Y"), true,
                    "Input(Y) of SequencePadGradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                    "Input(W) of SequencePadGradOp should not be null.");
  PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                    "Input(Out@GRAD) of SequencePadGradOp should not be null.");

  if (ctx->HasOutput(framework::GradVarName("X"))) {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
  }
  if (ctx->HasOutput(framework::GradVarName("Y"))) {
    ctx->SetOutputDim(framework::GradVarName("Y"), ctx->GetInputDim("Y"));
    ctx->ShareLoD("Y", /*->*/ framework::GradVarName("Y"));
  }
  if (ctx->HasOutput(framework::GradVarName("W"))) {
    ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
  }
}

void MatchMatrixTensorOpMaker::Make() {
  AddInput("X",
           "X (LoDTensor, default LoDTensor<float>) Input variable which "
           "should contain lod information.");
  AddInput("Y",
           "Y (LoDTensor, default LoDTensor<float>) Input variable which "
           "should contain lod information.");
  AddInput("W", "W (Tensor), The weight of X and Y.");
  AddAttr<int>("dim_t", "the dim of W").SetDefault(1);
  AddOutput("Out",
            "(LoDTensor, default LoDTensor<float>) Output variable which "
            "is X * W * Y");
  AddOutput("Tmp",
            "(LoDTensor, default LoDTensor<float>) tmp variable which is "
            "used for X * W");
  AddComment(R"DOC(
      Match Matrix Tensor Operator

      This operator calculate X * W * Y, only support 2-D for X and Y.
      the output is a level-1 LodTensor: 
        level_0: dim_t
      
      NOTE: only support 'float32' data type now.

    )DOC");
}

template <typename DeviceContext, typename T>
class CPUMatchMatrixTensorOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* y = ctx.Input<LoDTensor>("Y");
    auto* w = ctx.Input<Tensor>("W");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* tmp = ctx.Output<LoDTensor>("Tmp");

    int dim_t = ctx.Attr<int>("dim_t");
    int64_t dim_in = x->dims()[1];

    const auto& offset_l = x->lod()[0];
    const auto& offset_r = y->lod()[0];

    std::vector<size_t> top_offset;
    size_t top_size = 0;
    top_offset.push_back(top_size);
    for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
      size_t len_l = offset_l[b + 1] - offset_l[b];
      size_t len_r = offset_r[b + 1] - offset_r[b];
      top_size += dim_t * len_l * len_r;
      top_offset.push_back(top_size);
    }
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    memset(out_data, 0.0, out->dims()[0] * out->dims()[1] * sizeof(T));

    auto* bottom_l_data = x->data<T>();
    auto* bottom_r_data = y->data<T>();
    auto* t_data = w->data<T>();
    auto* bottom_l_trans_data = tmp->mutable_data<T>(ctx.GetPlace());
    memset(bottom_l_trans_data, 0.0,
           tmp->dims()[0] * tmp->dims()[1] * sizeof(T));

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

    call_gemm(blas, CblasNoTrans, CblasNoTrans, x->dims()[0], dim_t * dim_in,
              dim_in, 1.0f, bottom_l_data, t_data, 0.0f, bottom_l_trans_data);

    for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
      for (int t = 0; t < dim_t; t++) {
        size_t len_l = offset_l[b + 1] - offset_l[b];
        size_t len_r = offset_r[b + 1] - offset_r[b];
        auto* top_data = out_data + top_offset[b] + t * len_l * len_r;
        const auto* l_t_data =
            bottom_l_trans_data + offset_l[b] * dim_t * dim_in + t * dim_in;
        const auto* r_data = bottom_r_data + offset_r[b] * dim_in;
        auto blas_2 = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
        call_gemm_with_lda(blas_2, CblasNoTrans, CblasTrans, len_l, len_r,
                           dim_in, 1.0f, l_t_data, r_data, 0.0f, top_data,
                           dim_t * dim_in);
      }
    }

    framework::LoD out_lod;
    out_lod.push_back(top_offset);

    out->set_lod(out_lod);
  }
};

template <typename DeviceContext, typename T>
class CPUMatchMatrixTensorOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<LoDTensor>("X");
    auto* y = ctx.Input<LoDTensor>("Y");
    auto* w = ctx.Input<Tensor>("W");
    auto* tmp = ctx.Input<LoDTensor>("Tmp");

    int dim_t = ctx.Attr<int>("dim_t");
    int64_t dim_in = x->dims()[1];

    const auto& offset_l = x->lod()[0];
    const auto& offset_r = y->lod()[0];
    std::vector<size_t> top_offset;
    size_t top_size = 0;
    top_offset.push_back(top_size);
    for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
      size_t len_l = offset_l[b + 1] - offset_l[b];
      size_t len_r = offset_r[b + 1] - offset_r[b];
      top_size += dim_t * len_l * len_r;
      top_offset.push_back(top_size);
    }

    auto* bottom_l_data = x->data<T>();
    auto* bottom_r_data = y->data<T>();
    auto* bottom_l_trans_data = tmp->data<T>();

    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Output<LoDTensor>(framework::GradVarName("Y"));

    Tensor tmp_grad;
    tmp_grad.Resize(tmp->dims());
    auto* d_tmp_data = tmp_grad.mutable_data<T>(ctx.GetPlace());
    auto* top_diff = d_out->data<T>();
    auto* bottom_l_diff = d_x->mutable_data<T>(ctx.GetPlace());
    auto* bottom_r_diff = d_y->mutable_data<T>(ctx.GetPlace());
    auto* bottom_l_trans_diff = const_cast<T*>(d_tmp_data);
    memset(bottom_l_diff, 0.0, x->dims()[0] * x->dims()[1] * sizeof(T));
    memset(bottom_r_diff, 0.0, y->dims()[0] * y->dims()[1] * sizeof(T));
    memset(bottom_l_trans_diff, 0.0,
           tmp->dims()[0] * tmp->dims()[1] * sizeof(T));

    for (size_t b = 0; b < x->lod()[0].size() - 1; b++) {
      for (int t = 0; t < dim_t; t++) {
        size_t len_l = offset_l[b + 1] - offset_l[b];
        size_t len_r = offset_r[b + 1] - offset_r[b];

        for (size_t i = 0; i < len_l; i++) {
          for (size_t j = 0; j < len_r; j++) {
            auto diff =
                top_diff[top_offset[b] + t * len_l * len_r + i * len_r + j];
            auto* l_trans_data = bottom_l_trans_data +
                                 (offset_l[b] + i) * dim_in * dim_t +
                                 t * dim_in;
            auto* l_trans_diff = bottom_l_trans_diff +
                                 (offset_l[b] + i) * dim_in * dim_t +
                                 t * dim_in;
            auto* r_data = bottom_r_data + (offset_r[b] + j) * dim_in;
            auto* r_diff = bottom_r_diff + (offset_r[b] + j) * dim_in;
            if (diff != 0.0) {
              avx_axpy(r_data, l_trans_diff, dim_in, diff);
              avx_axpy(l_trans_data, r_diff, dim_in, diff);
            }
          }
        }
      }
    }

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

    auto* t_data = w->data<T>();
    auto* d_w = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto* t_diff = d_w->mutable_data<T>(ctx.GetPlace());
    memset(t_diff, 0.0, w->dims()[0] * w->dims()[1] * w->dims()[2] * sizeof(T));
    // bottom_diff
    call_gemm(blas, CblasNoTrans, CblasTrans, x->dims()[0], dim_in,
              dim_t * dim_in, 1.0f, bottom_l_trans_diff, t_data, 1.0f,
              bottom_l_diff);

    // t_diff
    call_gemm(blas, CblasTrans, CblasNoTrans, dim_in, dim_t * dim_in,
              x->dims()[0], 1.0f, bottom_l_data, bottom_l_trans_diff, 1.0f,
              t_diff);
  }
};

template <typename T>
class MatchMatrixTensorGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    auto* grad_op = new T();
    grad_op->SetType("match_matrix_tensor_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Y", this->Input("Y"));
    grad_op->SetInput("W", this->Input("W"));
    grad_op->SetInput("Tmp", this->Output("Tmp"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    grad_op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    grad_op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    match_matrix_tensor, ops::MatchMatrixTensorOP,
    ops::MatchMatrixTensorOpMaker,
    ops::MatchMatrixTensorGradOpMaker<paddle::framework::OpDesc>,
    ops::MatchMatrixTensorGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(match_matrix_tensor_grad, ops::MatchMatrixTensorOpGrad);

REGISTER_OP_CPU_KERNEL(match_matrix_tensor,
                       ops::CPUMatchMatrixTensorOPKernel<
                           paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_CPU_KERNEL(match_matrix_tensor_grad,
                       ops::CPUMatchMatrixTensorOPGradKernel<
                           paddle::platform::CPUDeviceContext, float>);

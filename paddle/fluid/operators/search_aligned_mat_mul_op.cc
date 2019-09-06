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
#include <vector>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;
using DDim = framework::DDim;

void assign_dims(int64_t x_dims_1, int64_t x_aligned_size, int64_t y_dims_1,
                 int64_t y_aligned_size, CBLAS_TRANSPOSE trans_x,
                 CBLAS_TRANSPOSE trans_y, std::vector<int64_t>& _dims) {
  std::vector<CBLAS_TRANSPOSE> _trans{trans_x, trans_y};
  _dims.resize(3);

  const auto bot0_aligned_size = x_aligned_size;
  const auto bot1_aligned_size = y_aligned_size;

  _dims[0] = (_trans[0] == CblasTrans) ? x_dims_1 : bot0_aligned_size;
  _dims[1] = (_trans[0] == CblasTrans) ? bot0_aligned_size : x_dims_1;
  _dims[2] = (_trans[1] == CblasTrans) ? bot1_aligned_size : y_dims_1;

  int bot1_row_num = (_trans[1] == CblasTrans) ? y_dims_1 : bot1_aligned_size;
  PADDLE_ENFORCE_EQ(_dims[1], bot1_row_num,
                    "Mismatch size, bot0_final_cols=[%d] bot1_final_rows=[%d]",
                    _dims[1], bot1_row_num);
}

void assign_dims(const DDim& x_dims, const LoD& x_lod, const DDim& y_dims,
                 const LoD& y_lod, CBLAS_TRANSPOSE trans_x, CBLAS_TRANSPOSE trans_y,
                 std::vector<int64_t>& _dims) {

  std::vector<CBLAS_TRANSPOSE> _trans{trans_x, trans_y};
  _dims.resize(3);

  const auto bot0_aligned_size = static_cast<int64_t>(x_lod[0][1]);
  const auto bot1_aligned_size = static_cast<int64_t>(y_lod[0][1]);

  _dims[0] = (_trans[0] == CblasTrans) ? x_dims[1] : bot0_aligned_size;
  _dims[1] = (_trans[0] == CblasTrans) ? bot0_aligned_size : x_dims[1];
  _dims[2] = (_trans[1] == CblasTrans) ? bot1_aligned_size : y_dims[1];

  int bot1_row_num = (_trans[1] == CblasTrans) ? y_dims[1] : bot1_aligned_size;
  PADDLE_ENFORCE_EQ(_dims[1], bot1_row_num,
         "Mismatch size, bot0_final_cols=[%d] bot1_final_rows=[%d]", _dims[1],
         bot1_row_num);
}

class SearchAlignedMatMulOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("Y",
             "Y (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");

    AddAttr<bool>("transpose_X", "If true, use the transpose of `X`.")
        .SetDefault(false);
    AddAttr<bool>("transpose_Y", "If true, use the transpose of `Y`.")
        .SetDefault(false);
    AddAttr<float>("alpha", "The scale of Out").SetDefault(1.0f);

    AddOutput("Out", "Out (Tensor, default Tensor<float>) Output variable");
    AddOutput("_a_addr",
              "_a_addr (Tensor, default Tensor<float>) Output variable");
    AddOutput("_b_addr",
              "_b_addr (Tensor, default Tensor<float>) Output variable");
    AddOutput("_c_addr",
              "_c_addr (Tensor, default Tensor<float>) Output variable");

    AddComment(R"DOC(
  SearchAlignedMatMul

  NOTE: only support 'float32' data type now.

)DOC");
  }
};

class SearchAlignedMatMulOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Y(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("_a_addr"),
                   "_a_addr(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("_b_addr"),
                   "_b_addr(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("_c_addr"),
                   "_c_addr(Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "X should be 2-D tensor");

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
      bool trans_x = ctx->Attrs().Get<bool>("transpose_X");
      bool trans_y = ctx->Attrs().Get<bool>("transpose_Y");

      std::vector<CBLAS_TRANSPOSE> _trans{CblasNoTrans, CblasNoTrans};
      _trans[0] = trans_x ? CblasTrans : CblasNoTrans;
      _trans[1] = trans_y ? CblasTrans : CblasNoTrans;

      std::vector<int64_t> _dims;
      assign_dims(x_dims[1], -1, y_dims[1], -1, _trans[0], _trans[1], _dims);
      ctx->SetOutputDim("Out", framework::make_ddim({-1, _dims[2]}));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchAlignedMatMulOPKernel : public framework::OpKernel<T> {
 public:
  void prepare_ff(const framework::ExecutionContext& ctx,
                  std::vector<int64_t>& _dims) const {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* _a_addr = ctx.Output<Tensor>("_a_addr");
    auto* _b_addr = ctx.Output<Tensor>("_b_addr");
    auto* _c_addr = ctx.Output<Tensor>("_c_addr");

    const int batch = bottom0->lod()[0].size() - 1;
    _a_addr->Resize(framework::make_ddim({batch}));
    _b_addr->Resize(framework::make_ddim({batch}));
    _c_addr->Resize(framework::make_ddim({batch}));

    T** a_addr_data = (T**)_a_addr->mutable_data<T>(ctx.GetPlace());
    T** b_addr_data = (T**)_b_addr->mutable_data<T>(ctx.GetPlace());
    T** c_addr_data = (T**)_c_addr->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_EQ(_dims.size(), 3, "_dims.size() should be eq 3.");
    const int bot0_size = _dims[0] * _dims[1];
    const int bot1_size = _dims[1] * _dims[2];
    const int top_size = _dims[0] * _dims[2];

    for (int i = 0; i < batch; ++i) {
      a_addr_data[i] = const_cast<T*>(bottom0->data<T>()) + bot0_size * i;
      b_addr_data[i] = const_cast<T*>(bottom1->data<T>()) + bot1_size * i;
      c_addr_data[i] = top->mutable_data<T>(ctx.GetPlace()) + top_size * i;
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* _a_addr = ctx.Output<Tensor>("_a_addr");
    auto* _b_addr = ctx.Output<Tensor>("_b_addr");
    auto* _c_addr = ctx.Output<Tensor>("_c_addr");
    float _scale = ctx.Attr<float>("alpha");
    
    bool trans_x = ctx.Attr<bool>("transpose_X");
    bool trans_y = ctx.Attr<bool>("transpose_Y");

    std::vector<CBLAS_TRANSPOSE> _trans{CblasNoTrans, CblasNoTrans};
    _trans[0] = trans_x ? CblasTrans : CblasNoTrans;
    _trans[1] = trans_y ? CblasTrans : CblasNoTrans;

    std::vector<int64_t> _dims;
    assign_dims(bottom0->dims(), bottom0->lod(), bottom1->dims(),
                bottom1->lod(), _trans[0], _trans[1], _dims);

    const int batch = bottom0->lod()[0].size() - 1;
    std::vector<size_t> offset(batch + 1);
    for (int i = 0; i <= batch; ++i) {
      offset[i] = _dims[0] * i;
    }

    framework::LoD top_lod;
    top_lod.push_back(offset);
    top->set_lod(top_lod);
    top->Resize(framework::make_ddim({static_cast<int64_t>(offset[batch]), _dims[2]}));

    prepare_ff(ctx, _dims);

    call_gemm_batched(ctx, _trans[0], _trans[1], static_cast<int>(_dims[0]), static_cast<int>(_dims[2]), static_cast<int>(_dims[1]),
                      _scale, (const T**)_a_addr->data<T>(),
                      (const T**)_b_addr->data<T>(), (T)0.0,
                      (T**)_c_addr->mutable_data<T>(ctx.GetPlace()), batch);
  }
};

class SearchAlignedMatMulOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("_a_addr"),
                   "_a_addr(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("_b_addr"),
                   "_b_addr(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("_c_addr"),
                   "_c_addr(Output) should not be null.");

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
class CPUSearchAlignedMatMulOPGradKernel : public framework::OpKernel<T> {
 public:
  void prepare_bp(const framework::ExecutionContext& ctx,
                  std::vector<int64_t>& _dims, Tensor& _a_addr_diff,
                  Tensor& _b_addr_diff, Tensor& _c_addr_diff) const {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* _a_addr = ctx.Input<Tensor>("_a_addr");
    auto* _b_addr = ctx.Input<Tensor>("_b_addr");
    auto* _c_addr = ctx.Input<Tensor>("_c_addr");
    auto* d_out = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Output<LoDTensor>(framework::GradVarName("Y"));

    const int batch = bottom0->lod()[0].size() - 1;
    PADDLE_ENFORCE_EQ(_a_addr->dims()[0], batch, "blob should be initialized before bp");

    _a_addr_diff.Resize(_a_addr->dims());
    _b_addr_diff.Resize(_b_addr->dims());
    _c_addr_diff.Resize(_c_addr->dims());
    T** a_addr_diff = (T**)_a_addr_diff.mutable_data<T>(ctx.GetPlace());
    T** b_addr_diff = (T**)_b_addr_diff.mutable_data<T>(ctx.GetPlace());
    T** c_addr_diff = (T**)_c_addr_diff.mutable_data<T>(ctx.GetPlace());

    

    const int bot0_size = _dims[0] * _dims[1];
    const int bot1_size = _dims[1] * _dims[2];
    const int top_size = _dims[0] * _dims[2];

    for (int i = 0; i < batch; ++i) {
      a_addr_diff[i] = d_x->mutable_data<T>(ctx.GetPlace()) + bot0_size * i;
      b_addr_diff[i] = d_y->mutable_data<T>(ctx.GetPlace()) + bot1_size * i;
      c_addr_diff[i] =
          const_cast<T*>(d_out->data<T>()) + top_size * i;
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<LoDTensor>("X");
    auto* bottom1 = ctx.Input<LoDTensor>("Y");
    auto* _a_addr = ctx.Input<Tensor>("_a_addr");
    auto* _b_addr = ctx.Input<Tensor>("_b_addr");
    bool trans_x = ctx.Attr<bool>("transpose_X");
    bool trans_y = ctx.Attr<bool>("transpose_Y");
    float _scale = ctx.Attr<float>("alpha");

    std::vector<CBLAS_TRANSPOSE> _trans{CblasNoTrans, CblasNoTrans};
    _trans[0] = trans_x ? CblasTrans : CblasNoTrans;
    _trans[1] = trans_y ? CblasTrans : CblasNoTrans;

    std::vector<int64_t> _dims;
    assign_dims(bottom0->dims(), bottom0->lod(), bottom1->dims(),
                bottom1->lod(), _trans[0], _trans[1], _dims);

    Tensor _a_addr_diff, _b_addr_diff, _c_addr_diff;
    prepare_bp(ctx, _dims, _a_addr_diff, _b_addr_diff, _c_addr_diff);

    const int batch = bottom0->lod()[0].size() - 1;
    if (_trans[1] == CblasTrans) {
      call_gemm_batched(
          ctx, CblasTrans, _trans[0], _dims[2], _dims[1], _dims[0], _scale,
          (const T**)_c_addr_diff.data<T>(), (const T**)_a_addr->data<T>(),
          (T)0.0, (T**)_b_addr_diff.mutable_data<T>(ctx.GetPlace()), batch);
    } else {
      CBLAS_TRANSPOSE bot0_trans =
          _trans[0] == CblasTrans ? CblasNoTrans : CblasTrans;
      call_gemm_batched(
          ctx, bot0_trans, CblasNoTrans, static_cast<int>(_dims[1]), static_cast<int>(_dims[2]), static_cast<int>(_dims[0]), _scale,
          (const T**)_a_addr->data<T>(), (const T**)_c_addr_diff.data<T>(),
          (T)0.0, (T**)_b_addr_diff.mutable_data<T>(ctx.GetPlace()), batch);
    }

    if (_trans[0] == CblasTrans) {
      call_gemm_batched(
          ctx, _trans[1], CblasTrans, _dims[1], _dims[0], _dims[2], _scale,
          (const T**)_b_addr->data<T>(), (const T**)_c_addr_diff.data<T>(),
          (T)0.0, (T**)_a_addr_diff.mutable_data<T>(ctx.GetPlace()), batch);
    } else {
      CBLAS_TRANSPOSE bot1_trans =
          (_trans[1] == CblasTrans) ? CblasNoTrans : CblasTrans;
      call_gemm_batched(
          ctx, CblasNoTrans, bot1_trans, _dims[0], _dims[1], _dims[2], _scale,
          (const T**)_c_addr_diff.data<T>(), (const T**)_b_addr->data<T>(),
          (T)0.0, (T**)_a_addr_diff.mutable_data<T>(ctx.GetPlace()), batch);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_aligned_mat_mul, ops::SearchAlignedMatMulOP,
                  ops::SearchAlignedMatMulOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_aligned_mat_mul_grad, ops::SearchAlignedMatMulOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_aligned_mat_mul,
    ops::CPUSearchAlignedMatMulOPKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchAlignedMatMulOPKernel<plt::CPUDeviceContext,
    //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_aligned_mat_mul_grad,
    ops::CPUSearchAlignedMatMulOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchAlignedMatMulOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

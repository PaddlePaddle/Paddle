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
#include "naive_gemm.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

template <typename DeviceContext, typename T>
void call_gemm(const math::BlasT<DeviceContext, T>& blas,
               const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
               const int M, const int N, const int K, const T alpha, const T* A,
               const T* B, const T beta, T* C, bool navie) {
if (!navie) {
  VLOG(1) << "use normal gemm";
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  blas.GEMM(TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
} else {
  VLOG(1) << "use naive gemm";
  naive::gemm((TransA == CblasTrans), (TransB == CblasTrans), M, N, K, alpha, A,
              B, beta, C);
}
}

// To align with Lego
#ifndef LEGO_USE_FLOAT
#define LEGO_USE_FLOAT
#endif
#ifndef LEGO_SSE
#define LEGO_SSE
#endif

#if defined(LEGO_USE_FLOAT)

#define __m256x __m256
#define __m128x __m128

static const unsigned int AVX_STEP_SIZE = 8;
static const unsigned int SSE_STEP_SIZE = 4;
static const unsigned int AVX_CUT_LEN_MASK = 7U;
static const unsigned int SSE_CUT_LEN_MASK = 3U;

#define _mm256_setzero_px _mm256_setzero_ps
#define _mm256_mul_px _mm256_mul_ps
#define _mm256_add_px _mm256_add_ps
#define _mm256_load_px _mm256_loadu_ps
#define _mm256_hadd_px _mm256_hadd_ps
#define _mm256_permute2f128_px _mm256_permute2f128_ps
#define _mm256_store_px _mm256_storeu_ps
#define _mm256_broadcast_sx _mm256_broadcast_ss
#define _mm256_castpx256_px128 _mm256_castps256_ps128
#define _mm256_max_px _mm256_max_ps
#define _mm256_sub_px _mm256_sub_ps
#define _mm256_set1_px _mm256_set1_ps
#define _mm256_sqrt_px _mm256_sqrt_ps
#define _mm256_div_px _mm256_div_ps
#define _mm_setzero_px _mm_setzero_ps
#define _mm_add_px _mm_add_ps
#define _mm_mul_px _mm_mul_ps
#define _mm_load_px _mm_loadu_ps
#define _mm_hadd_px _mm_hadd_ps
#define _mm_store_sx _mm_store_ss
#define _mm_store_px _mm_storeu_ps
#define _mm_load1_px _mm_load1_ps
#define _mm_max_px _mm_max_ps
#define _mm_sub_px _mm_sub_ps
#define _mm_set1_px _mm_set1_ps
#define _mm_sqrt_px _mm_sqrt_ps
#define _mm_div_px _mm_div_ps

#elif defined(LEGO_USE_DOUBLE)

#define __m256x __m256d
#define __m128x __m128d

static const unsigned int AVX_STEP_SIZE = 4;
static const unsigned int SSE_STEP_SIZE = 2;
static const unsigned int AVX_CUT_LEN_MASK = 3U;
static const unsigned int SSE_CUT_LEN_MASK = 1U;

#define _mm256_setzero_px _mm256_setzero_pd
#define _mm256_mul_px _mm256_mul_pd
#define _mm256_add_px _mm256_add_pd
#define _mm256_load_px _mm256_loadu_pd
#define _mm256_hadd_px _mm256_hadd_pd
#define _mm256_permute2f128_px _mm256_permute2f128_pd
#define _mm256_store_px _mm256_storeu_pd
#define _mm256_broadcast_sx _mm256_broadcast_sd
#define _mm256_castpx256_px128 _mm256_castpd256_pd128
#define _mm256_max_px _mm256_max_pd
#define _mm256_sub_px _mm256_sub_pd
#define _mm256_set1_px _mm256_set1_pd
#define _mm256_sqrt_px _mm256_sqrt_pd
#define _mm256_div_px _mm256_div_pd
#define _mm_setzero_px _mm_setzero_pd
#define _mm_add_px _mm_add_pd
#define _mm_mul_px _mm_mul_pd
#define _mm_load_px _mm_loadu_pd
#define _mm_hadd_px _mm_hadd_pd
#define _mm_store_sx _mm_store_sd
#define _mm_store_px _mm_storeu_pd
#define _mm_load1_px _mm_load1_pd
#define _mm_max_px _mm_max_pd
#define _mm_sub_px _mm_sub_pd
#define _mm_set1_px _mm_set1_pd
#define _mm_sqrt_px _mm_sqrt_pd
#define _mm_div_px _mm_div_pd
#endif

template <typename T>
inline void sse_eltadd(const T* x, const T* y, T* z, size_t len) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(z + jjj, _mm256_add_px(_mm256_load_px(x + jjj),
                                           _mm256_load_px(y + jjj)));
  }
#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;

  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(z + jjj,
                 _mm_add_px(_mm_load_px(x + jjj), _mm_load_px(y + jjj)));
  }
#endif
  for (; jjj < len; jjj++) {
    z[jjj] = x[jjj] + y[jjj];
  }
}

class SearchFCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (Tensor, default Tensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("W", "W (Tensor)");
    AddInput("b", "b (Tensor)");
    AddAttr<int>("out_size", "out_size: the output size")
        .SetDefault(0)
        .EqualGreaterThan(1);
    AddAttr<bool>("navie", "navie: use navie gemm")
        .SetDefault(false);

    AddOutput("Out", "Out (Tensor, default Tensor<float>) Output variable");

    AddComment(R"DOC(
      SearchFC
      
      NOTE: only support 'float32' data type now.

    )DOC");
  }
};

class SearchFCOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "W(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("b"), "b(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2, "W should be 2-D tensor");

    auto b_dims = ctx->GetInputDim("b");
    PADDLE_ENFORCE_EQ(b_dims.size(), 1, "b should be 1-D tensor");

    int out_size = ctx->Attrs().Get<int>("out_size");

    ctx->SetOutputDim("Out", framework::make_ddim({-1, out_size}));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(w_dims[1], x_dims[1], "wrong shape: w_dims[1] != x_dims[1]");
    }
    else {
      // compile time
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchFCOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<Tensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    auto* b = ctx.Input<Tensor>("b");
    auto* top = ctx.Output<Tensor>("Out");

    int out_size = ctx.Attr<int>("out_size");  // 100
    bool navie = ctx.Attr<bool>("navie");  // 100
    int batch = bottom->dims()[0];

    int _out = w->dims()[0];  // 100
    int _in = w->dims()[1];   // 228
    //PADDLE_ENFORCE_EQ(out_size, _out, "out_size should equal to w->dims()[1]");
    //PADDLE_ENFORCE_EQ(bottom->dims()[1], _in,
    //                  "x.dims()[1] should equal to w->dims()[0]");

    top->Resize(framework::make_ddim({bottom->dims()[0], out_size}));

    const auto* bottom_data = bottom->data<T>();
    auto* top_data = top->mutable_data<T>(ctx.GetPlace());
    const auto* weights = w->data<T>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    call_gemm(blas, CblasNoTrans, CblasTrans, batch, _out, _in, 1.0f,
              bottom_data, weights, 0.0f, top_data, navie);
    if (true) {
      const auto* bias_data = b->data<T>();
      for (int i = 0; i < batch; ++i) {
        // add bias here
        sse_eltadd(top_data + i * _out, bias_data, top_data + i * _out, _out);
      }
    }
  }
};

class SearchFCOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("b"), "Input(b) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) of SequencePadGradOp should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("W"))) {
      ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
    }
    if (ctx->HasOutput(framework::GradVarName("b"))) {
      ctx->SetOutputDim(framework::GradVarName("b"), ctx->GetInputDim("b"));
    }
  }
};

template <typename DeviceContext, typename T>
class CPUSearchFCOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //     auto* d_x = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    //
    //     auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());
    //     auto* x = ctx.Input<LoDTensor>("X");
    //     memset(bottom_diff, 0.0, x->dims()[0] * x->dims()[1] * sizeof(T));
    auto* bottom = ctx.Input<Tensor>("X");
    auto* w = ctx.Input<Tensor>("W");
    int _out = w->dims()[0];  // 100
    int _in = w->dims()[1];   // 228

    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_w = ctx.Output<Tensor>(framework::GradVarName("W"));
    bool navie = ctx.Attr<bool>("navie");  // 100

    int batch = bottom->dims()[0];
    const auto* top_diff = d_out->data<T>();
    const auto* bottom_data = bottom->data<T>();
    auto* bottom_diff = d_x->mutable_data<T>(ctx.GetPlace());

    const auto* weights = w->data<T>();
    auto* weights_diff = d_w->mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    //call_gemm(blas, CblasTrans, CblasNoTrans, _in, _out, batch, 1.0f,
    //          bottom_data, top_diff, 0.0f, weights_diff);
    call_gemm(blas, CblasTrans, CblasNoTrans, _out, _in, batch, (T)1.0,
                          top_diff, bottom_data, (T)0.0, weights_diff, navie);

    call_gemm(blas, CblasNoTrans, CblasNoTrans, batch, _in, _out, (T)1.0, top_diff,
              weights, (T)0.0, bottom_diff, navie);

    if (true) {
      auto* d_b = ctx.Output<Tensor>(framework::GradVarName("b"));
      auto* bias_diff = d_b->mutable_data<T>(ctx.GetPlace());
      memset(bias_diff, 0.0, _out * sizeof(T));
      for (int i = 0; i < batch; ++i) {
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
REGISTER_OPERATOR(search_fc, ops::SearchFCOP, ops::SearchFCOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_fc_grad, ops::SearchFCOpGrad);

REGISTER_OP_CPU_KERNEL(search_fc,
                       ops::CPUSearchFCOPKernel<plt::CPUDeviceContext, float>
                       //     ops::CPUSearchFCOPKernel<plt::CPUDeviceContext,
                       //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_fc_grad, ops::CPUSearchFCOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchFCOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

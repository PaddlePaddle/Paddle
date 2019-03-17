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
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

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

template<class DTYPE>
inline void sse_ip(const DTYPE* vec1, const DTYPE* vec2, size_t len, DTYPE& result){
    unsigned int jjj, lll; 
    jjj = lll = 0;
    result = 0.;

#if defined(LEGO_AVX) 
    lll = len&~AVX_CUT_LEN_MASK;

    __m256x mm_result = _mm256_setzero_px(); 
    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE){ 

        mm_result = _mm256_add_px(mm_result, 
          _mm256_mul_px( 
              _mm256_load_px(vec1+jjj), 
              _mm256_load_px(vec2+jjj) 
              )); 
    } 

    //    result = mm_result[0]+mm_result[1]+mm_result[2]+mm_result[3]+
    //      mm_result[4]+mm_result[5]+mm_result[6]+mm_result[7];

#if defined(LEGO_USE_FLOAT)
    __m256x hsum = _mm256_hadd_px(mm_result, mm_result);
#elif defined(LEGO_USE_DOUBLE)
    __m256x hsum = mm_result; 
#endif

    hsum = _mm256_add_px(hsum, _mm256_permute2f128_px(hsum, hsum, 0x1));

    _mm_store_sx(&result, 
      _mm_hadd_px( 
          _mm256_castpx256_px128(hsum), _mm256_castpx256_px128(hsum) 
          ));

#elif defined(LEGO_SSE) 
    lll = len&~SSE_CUT_LEN_MASK; 
    __m128x mm_result = _mm_setzero_px(); 
    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE){ 
        mm_result = _mm_add_px(mm_result, 
          _mm_mul_px( 
              _mm_load_px(vec1+jjj), 
              _mm_load_px(vec2+jjj) 
              )); 
    } 
    __m128x mm_tmp = _mm_hadd_px(mm_result, mm_result); 
#if defined(LEGO_USE_FLOAT)
    _mm_store_sx(&result, _mm_hadd_px(mm_tmp, mm_tmp)); 
#elif defined(LEGO_USE_DOUBLE)
    _mm_store_sx(&result, mm_tmp); 
#endif

#endif
    for (; jjj<len; jjj++){
        result += vec1[jjj] * vec2[jjj];
    }
}

template<class DTYPE>
inline void sse_scale(const DTYPE* x, DTYPE* y, size_t len, const DTYPE alpha) {
    unsigned int jjj, lll; 
    jjj = lll = 0;

#if defined(LEGO_AVX) 
    lll = len&~AVX_CUT_LEN_MASK; 
    __m256x mm_alpha = _mm256_broadcast_sx(&alpha); 

    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE){ 
        _mm256_store_px(y+jjj, 
          _mm256_mul_px(mm_alpha, 
              _mm256_load_px(x+jjj))); 
    }

#elif defined(LEGO_SSE) 
    lll = len&~SSE_CUT_LEN_MASK; 
    __m128x mm_alpha = _mm_load1_px(&alpha); 
    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE){ 
        _mm_store_px(y+jjj, 
          _mm_mul_px(mm_alpha, 
              _mm_load_px(x+jjj))); 
    } 
#endif
    for (; jjj < len; jjj++) { 
        y[jjj] = alpha * x[jjj]; 
    } 
}

template<class DTYPE>
inline void sse_axpy(const DTYPE* x, DTYPE* y, size_t len, const DTYPE alpha) {
    unsigned int jjj, lll; 
    jjj = lll = 0;

#if defined(LEGO_AVX) 
    lll = len&~AVX_CUT_LEN_MASK; 
    __m256x mm_alpha = _mm256_broadcast_sx(&alpha); 
    for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE){ 
        _mm256_store_px(y+jjj, 
          _mm256_add_px(_mm256_load_px(y+jjj), 
              _mm256_mul_px(mm_alpha, 
                  _mm256_load_px(x+jjj)))); 
    }

#elif defined(LEGO_SSE) 
    lll = len&~SSE_CUT_LEN_MASK; 
    __m128x mm_alpha = _mm_load1_px(&alpha); 
    for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE){ 
        _mm_store_px(y+jjj, 
          _mm_add_px(_mm_load_px(y+jjj), 
              _mm_mul_px(mm_alpha, 
                  _mm_load_px(x+jjj)))); 
    }

#endif
    for (; jjj<len; jjj++){
        y[jjj] += alpha * x[jjj];
    }
}

class SearchCosineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The 1st input of cos_sim op.");
    AddInput("Y", "The 2nd input of cos_sim op.");
    AddOutput("Out", "The output of cos_sim op.");
    AddOutput("XX",
              "inner product of the first input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();
    AddOutput("YY",
              "inner product of the second input, reduced along the 1st "
              "dimension.")
        .AsIntermediate();
    AddOutput("XY",
              "inner product of the inputs, reduced along the 1st "
              "dimension.")
        .AsIntermediate();

    AddComment(R"DOC(
**Cosine Similarity Operator**

$Out = \frac{X^T * Y}{(\sqrt{X^T * X} * \sqrt{Y^T * Y})}$

The input X and Y must have the same shape, except that the 1st dimension
of input Y could be just 1 (different from input X), which will be
broadcasted to match the shape of input X before computing their cosine
similarity.

Both the input X and Y can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.

)DOC");
  }
};

class SearchCosineOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // notnull check
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of CosSimOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of CosSimOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of CosSimOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("XX"),
                   "Output(XX) of CosSimOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("YY"),
                   "Output(YY) of CosSimOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("XY"),
                   "Output(XY) of CosSimOp should not be null.");

    // shape check
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dims.size(), y_dims.size(),
                      "Ranks of Input(X) and Input(Y) must be equal.");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) must not be less than 2.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 1, x_dims.size()),
                      framework::slice_ddim(y_dims, 1, y_dims.size()),
                      "All dimensions except the 1st of Input(X) and Input(Y) "
                      "must be equal.");
    PADDLE_ENFORCE(x_dims[0] == y_dims[0] || y_dims[0] == 1,
                   "The 1st dimension of Input(Y) must be equal to Input(X) or"
                   " just 1 (which will be broadcasted to match Input(X)).");

    // resize tensor
    ctx->SetOutputDim("Out", {x_dims[0], 1});
    ctx->SetOutputDim("XX", {x_dims[0], 1});
    ctx->SetOutputDim("YY", {y_dims[0], 1});
    ctx->SetOutputDim("XY", {y_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

static float _epsilon = 1e-10;
template <typename DeviceContext, typename DTYPE>
class CPUSearchCosineOPKernel : public framework::OpKernel<DTYPE> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<Tensor>("X");
    auto* bottom1 = ctx.Input<Tensor>("Y");
    auto* top = ctx.Output<Tensor>("Out");
    auto* x_x = ctx.Output<Tensor>("XX");
    auto* y_y = ctx.Output<Tensor>("YY");
    auto* x_y = ctx.Output<Tensor>("XY");

    int batch = bottom0->dims()[0];
    int vec_dim = bottom0->dims()[1];
    top->Resize(framework::make_ddim({batch, 1}));
    x_x->Resize(framework::make_ddim({batch, 1}));
    y_y->Resize(framework::make_ddim({batch, 1}));
    x_y->Resize(framework::make_ddim({batch, 1}));

    DTYPE * aa = x_y->mutable_data<DTYPE>(ctx.GetPlace());
    DTYPE * bb = x_x->mutable_data<DTYPE>(ctx.GetPlace());
    DTYPE * cc = y_y->mutable_data<DTYPE>(ctx.GetPlace());
    const DTYPE* bottom0_data = bottom0->data<DTYPE>();
    const DTYPE* bottom1_data = bottom1->data<DTYPE>();
    DTYPE * top_data = top->mutable_data<DTYPE>(ctx.GetPlace());
    for (int i = 0; i < batch; ++i) {
        sse_ip(bottom0_data+i*vec_dim, bottom1_data+i*vec_dim, vec_dim, aa[i]);
        sse_ip(bottom0_data+i*vec_dim, bottom0_data+i*vec_dim, vec_dim, bb[i]);
        sse_ip(bottom1_data+i*vec_dim, bottom1_data+i*vec_dim, vec_dim, cc[i]);
        DTYPE bc = bb[i] * cc[i];
        if (bc < _epsilon) {
            top_data[i] = 0.0;
        } else {
            top_data[i] = aa[i] / (sqrt(bc));
        }
    }
  }
};

class SearchCosineOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    // notnull check
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("XX"), "Input(XX) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("YY"), "Input(YY) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("XY"), "Input(XY) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Input(Out) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) must not be null.");

    // shape check
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto xx_dims = ctx->GetInputDim("XX");
    auto yy_dims = ctx->GetInputDim("YY");
    auto xy_dims = ctx->GetInputDim("XY");
    auto out_dims = ctx->GetInputDim("Out");
    auto out_grad_dims = ctx->GetInputDim(framework::GradVarName("Out"));

    PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                      "Ranks of Input(X) and Input(Y) must be equal.");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "Rank of Input(X) must not be less than 2.");
    PADDLE_ENFORCE_EQ(framework::slice_ddim(x_dims, 1, x_dims.size()),
                      framework::slice_ddim(y_dims, 1, y_dims.size()),
                      "All dimensions except the 1st of Input(X) and Input(Y) "
                      "must be equal.");
    PADDLE_ENFORCE(x_dims[0] == y_dims[0] || y_dims[0] == 1,
                   "The 1st dimension of Input(Y) must be equal to Input(X) or"
                   " just 1 (which will be broadcasted to match Input(X)).");
    auto target_xx_dims = framework::make_ddim({x_dims[0], 1});
    auto target_yy_dims = framework::make_ddim({y_dims[0], 1});
    PADDLE_ENFORCE_EQ(xx_dims, target_xx_dims,
                      "Shape of Input(XX) must be [X.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(yy_dims, target_yy_dims,
                      "Shape of Input(YY) must be [Y.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(xy_dims, target_yy_dims,
                      "Shape of Input(XY) must be [Y.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(out_dims, target_xx_dims,
                      "Shape of Input(Out) must be [X.Dim(0), 1].");
    PADDLE_ENFORCE_EQ(out_grad_dims, target_xx_dims,
                      "Shape of Input(Out@Grad) must be [X.Dim(0), 1].");

    // resize tensor
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename DeviceContext, typename DTYPE>
class CPUSearchCosineOPGradKernel : public framework::OpKernel<DTYPE> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom0 = ctx.Input<Tensor>("X");
    auto* bottom1 = ctx.Input<Tensor>("Y");
    auto* x_x = ctx.Input<Tensor>("XX");
    auto* y_y = ctx.Input<Tensor>("YY");
    auto* x_y = ctx.Input<Tensor>("XY");

    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Output<Tensor>(framework::GradVarName("Y"));

    auto * bottom0_diff = d_x->mutable_data<DTYPE>(ctx.GetPlace());
    auto * bottom1_diff = d_y->mutable_data<DTYPE>(ctx.GetPlace());

    int batch = bottom0->dims()[0];
    int vec_dim = bottom0->dims()[1];
   
    const DTYPE * aa = x_y->data<DTYPE>();
    const DTYPE * bb = x_x->data<DTYPE>();
    const DTYPE * cc = y_y->data<DTYPE>();
    const DTYPE* bottom0_data = bottom0->data<DTYPE>();
    const DTYPE* bottom1_data = bottom1->data<DTYPE>();
 
    const DTYPE * top_diff = d_out->data<DTYPE>();
    for (int i = 0; i < batch; ++i) {
        int offset = i * vec_dim;
        DTYPE a = aa[i];
        // adding a tiny value for numerical stability
        DTYPE b = 0.0;
        if (bb[i] > _epsilon) {
            b = DTYPE(1.0) / (sqrt(bb[i]) + _epsilon);
        } else {
            memset(bottom0_diff + offset, 0.0, vec_dim * sizeof(DTYPE));
            memset(bottom1_diff + offset, 0.0, vec_dim * sizeof(DTYPE));
            return;
        }
        
        DTYPE c = 0.0;
        if (cc[i] > _epsilon) {
            c = DTYPE(1.0) / (sqrt(cc[i]) + _epsilon);
        } else {
            memset(bottom0_diff + offset, 0.0, vec_dim * sizeof(DTYPE));
            memset(bottom1_diff + offset, 0.0, vec_dim * sizeof(DTYPE));
            return;
        }
        DTYPE t1 = c * b * top_diff[i];
        DTYPE t2 = -1 * a * c * b * b * b * top_diff[i];
        sse_scale(bottom1_data+offset, bottom0_diff+offset, vec_dim, t1);
        sse_axpy(bottom0_data+offset, bottom0_diff+offset, vec_dim, t2);
        t1 = -1 * a * b * c * c * c * top_diff[i];
        t2 = b * c * top_diff[i];
        sse_scale(bottom1_data+offset, bottom1_diff+offset, vec_dim, t1);
        sse_axpy(bottom0_data+offset, bottom1_diff+offset, vec_dim, t2);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_cosine, ops::SearchCosineOP, ops::SearchCosineOpMaker,
                  frm::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(search_cosine_grad, ops::SearchCosineOpGrad);

REGISTER_OP_CPU_KERNEL(search_cosine,
                       ops::CPUSearchCosineOPKernel<plt::CPUDeviceContext, float>
                       //     ops::CPUSearchCosineOPKernel<plt::CPUDeviceContext,
                       //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_cosine_grad, ops::CPUSearchCosineOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchCosineOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

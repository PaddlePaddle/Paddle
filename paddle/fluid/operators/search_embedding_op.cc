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

template <typename T>
inline void sse_axpy(const T* x, T* y, size_t len, const T alpha) {
  unsigned int jjj, lll;
  jjj = lll = 0;

#if defined(LEGO_AVX)
  lll = len & ~AVX_CUT_LEN_MASK;
  __m256x mm_alpha = _mm256_broadcast_sx(&alpha);
  for (jjj = 0; jjj < lll; jjj += AVX_STEP_SIZE) {
    _mm256_store_px(
        y + jjj,
        _mm256_add_px(_mm256_load_px(y + jjj),
                      _mm256_mul_px(mm_alpha, _mm256_load_px(x + jjj))));
  }

#elif defined(LEGO_SSE)
  lll = len & ~SSE_CUT_LEN_MASK;
  __m128x mm_alpha = _mm_load1_px(&alpha);
  for (jjj = 0; jjj < lll; jjj += SSE_STEP_SIZE) {
    _mm_store_px(y + jjj,
                 _mm_add_px(_mm_load_px(y + jjj),
                            _mm_mul_px(mm_alpha, _mm_load_px(x + jjj))));
  }

#endif
  for (; jjj < len; jjj++) {
    y[jjj] += alpha * x[jjj];
  }
}

class SearchEmbeddingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (Tensor, default Tensor<int_64>) Input variable which "
             "should contain lod information.");
    AddInput("W", "W (Tensor)");
    AddAttr<int>("num_voc", "num_voc").SetDefault(0).EqualGreaterThan(0);
    AddAttr<int>("num_emb", "num_emb").SetDefault(0).EqualGreaterThan(0);
    AddAttr<float>("lr", "learning rate").SetDefault(0.0).EqualGreaterThan(0.0);

    AddOutput("Out", "Out (Tensor, default Tensor<float>) Output variable");

    AddComment(R"DOC(
      SearchEmbedding
      
      NOTE: only support 'float32' data type now.

    )DOC");
  }
};

class SearchEmbeddingOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "W(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "The rank of X(Input) should be 2.");

    auto w_dims = ctx->GetInputDim("W");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2, "W should be 2-D tensor");

    int num_voc = ctx->Attrs().Get<int>("num_voc");
    int num_emb = ctx->Attrs().Get<int>("num_emb");

    PADDLE_ENFORCE_EQ(w_dims[0], num_voc,
                      "w_dims[0] should be equal to num_voc");
    PADDLE_ENFORCE_EQ(w_dims[1], num_emb,
                      "w_dims[1] should be equal to num_emb");

    ctx->SetOutputDim("Out", framework::make_ddim({-1, num_emb}));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class CPUSearchEmbeddingOPKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* _blobs = ctx.Input<Tensor>("W");
    auto* top = ctx.Output<LoDTensor>("Out");

    int _cap_e = ctx.Attr<int>("num_emb");

    int _cap_l = bottom->dims()[0];
    auto& offset = bottom->lod()[0];
    std::vector<size_t> top_offset;
    top_offset.resize(offset.size());
    top_offset[0] = 0;

    for (int i = 0; i < top_offset.size() - 1; ++i) {
      int w = offset[i + 1] - offset[i];
      if (w == 0) {
        top_offset[i + 1] = top_offset[i] + 1;
      } else {
        top_offset[i + 1] = top_offset[i] + w;
      }
    }

    int top_l = top_offset[top_offset.size() - 1];
    framework::LoD top_lod;
    top_lod.push_back(top_offset);
    top->set_lod(top_lod);
    top->Resize(framework::make_ddim({top_l, _cap_e}));

    PADDLE_ENFORCE_EQ(top_l, _cap_l,
                      "top_l should be equal to _cap_l");

    auto* top_data = top->mutable_data<T>(ctx.GetPlace());
    const auto* bottom_data = bottom->data<int64_t>();
    const auto* weights = _blobs->data<T>();

    for (int i = 0; i < offset.size() - 1; ++i) {
      int w = offset[i + 1] - offset[i];
      if (w == 1 && bottom_data[offset[i]] == -1) {
        //LOG (ERROR) << "zero len sequence " << i << "/" << top_offset.size() - 1;
        memset(top_data + top_offset[i] * _cap_e, 0, _cap_e * sizeof(T));
      } else {
        for (int j = 0; j < w; ++j) {
          unsigned int word_idx =
              static_cast<unsigned int>(bottom_data[offset[i] + j]);
          memcpy((void*)(top_data + (top_offset[i] + j) * _cap_e),
                 (void*)(weights + word_idx * _cap_e), _cap_e * sizeof(T));
        }
      }
    }
  }
};

class SearchEmbeddingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) of SearchEmbeddingGradOp should not be null.");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = framework::GetDataTypeOfVar(ctx.InputVar("W"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
  
};

class SearchEmbeddingGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op_desc_ptr = new framework::OpDesc();
    op_desc_ptr->SetType("search_embedding_grad");
    op_desc_ptr->SetInput("X", Input("X"));
    op_desc_ptr->SetInput("W", Input("W"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op_desc_ptr->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(op_desc_ptr);
  }
};

template <typename DeviceContext, typename T>
class CPUSearchEmbeddingOPGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* _blobs = ctx.Input<Tensor>("W");
    auto* top = ctx.Input<LoDTensor>(framework::GradVarName("Out"));

    int _cap_e = ctx.Attr<int>("num_emb");
    float _lr = ctx.Attr<float>("lr");

    auto& offset = bottom->lod()[0];
    auto& top_offset = top->lod()[0];

    const auto* top_diff = top->data<T>();
    const auto* bottom_data = bottom->data<int64_t>();
    T* weights = (T*) (_blobs->data<T>());

    T mlr = -1.0 * _lr;

    for (int i = 0; i < offset.size() - 1; ++i) {
      int w = offset[i + 1] - offset[i];
      if (!(w == 1 && bottom_data[offset[i]] == -1)) {
        for (int j = 0; j < w; ++j) {
          unsigned int word_idx =
              static_cast<unsigned int>(bottom_data[offset[i] + j]);
          sse_axpy((const T*)top_diff + (top_offset[i] + j) * _cap_e,
                   weights + word_idx * _cap_e, _cap_e, mlr);
        }
      } else {
        //LOG(ERROR) << "bp: zero len sequence " << i << "/"
        //           << top_offset.size() - 1;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_embedding, ops::SearchEmbeddingOP,
                  ops::SearchEmbeddingOpMaker, ops::SearchEmbeddingGradOpMaker);
REGISTER_OPERATOR(search_embedding_grad, ops::SearchEmbeddingOpGrad);

REGISTER_OP_CPU_KERNEL(search_embedding,
                       ops::CPUSearchEmbeddingOPKernel<plt::CPUDeviceContext, float>
                       //     ops::CPUSearchEmbeddingOPKernel<plt::CPUDeviceContext,
                       //                                       double>
);
REGISTER_OP_CPU_KERNEL(
    search_embedding_grad, ops::CPUSearchEmbeddingOPGradKernel<plt::CPUDeviceContext, float>
    //     ops::CPUSearchEmbeddingOPGradKernel<plt::CPUDeviceContext,
    //                                           double>
);

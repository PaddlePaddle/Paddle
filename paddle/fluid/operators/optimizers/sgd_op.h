/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/platform/dynload/mklml.h"

namespace paddle {
namespace operators {

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


template <typename DTYPE>
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
template <typename T>
class SGDOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param_var = ctx.InputVar("Param");
    const auto *grad_var = ctx.InputVar("Grad");

    if (param_var->IsType<framework::LoDTensor>()) {
      const auto *param = ctx.Input<framework::Tensor>("Param");
      auto *param_out = ctx.Output<framework::Tensor>("ParamOut");
      // Actually, all tensors are LoDTensor except SelectedRows.
      if (grad_var->IsType<framework::LoDTensor>()) {
        const auto *grad = ctx.Input<framework::Tensor>("Grad");
        auto sz = param_out->numel();
        PADDLE_ENFORCE_EQ(param->numel(), sz);
        PADDLE_ENFORCE_EQ(grad->numel(), sz);

        jit::sgd_attr_t attr(1, sz, 1, sz, 1);
        const T *lr = learning_rate->data<T>();
        const T *param_data = param->data<T>();
        const T *grad_data = grad->data<T>();
        int64_t rows_idx = 0;
        T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

        auto sgd =
            jit::KernelFuncs<jit::SgdTuple<T>, platform::CPUPlace>::Cache().At(
                attr);
        sgd(lr, param_data, grad_data, &rows_idx, out_data, &attr);
      } else if (grad_var->IsType<framework::SelectedRows>()) {
        // TODO(qijun): In Sparse SGD operator, in-place update is enforced.
        // This manual optimization brings difficulty to track data dependency.
        // It's better to find a more elegant solution.
        PADDLE_ENFORCE_EQ(param, param_out);
        const auto *grad = ctx.Input<framework::SelectedRows>("Grad");
        auto &grad_rows = grad->rows();

        // for distributed training, a sparse var may be empty,
        // just skip updating.
        if (grad_rows.size() == 0) {
          return;
        }

        auto out_dims = param_out->dims();
        PADDLE_ENFORCE_EQ(grad->height(), out_dims[0]);
        auto &grad_value = grad->value();
        //const T *param_data = param->data<T>();
        const T *grad_data = grad_value.data<T>();
        const T *lr = learning_rate->data<T>();
        const int64_t *rows_data = grad_rows.data();
        T *out_data = param_out->mutable_data<T>(ctx.GetPlace());
        auto grad_row_width = grad->value().dims()[1];
	
        for (size_t i = 0; i < grad->rows().size(); i++) {
	  int64_t id_index = rows_data[i];
	  PADDLE_ENFORCE_GE(id_index, static_cast<int64_t>(0),
			    "id should be in the table");
	  for (int64_t j = 0; j < grad_row_width; j++) {
	    out_data[id_index * grad_row_width + j] -=
		lr[0] * grad_data[i * grad_row_width + j];
	  }
	  //sse_axpy(&grad_data[i * grad_row_width], &out_data[id_index * grad_row_width], grad_row_width, static_cast<T>(-1*lr[0]));
	}

/*
        jit::sgd_attr_t attr;
        attr.param_height = out_dims[0];
        attr.param_width = param_out->numel() / attr.param_height;
        attr.grad_height = grad_rows.size();  // note: it is not grad->height()
        attr.grad_width = grad_value.numel() / attr.grad_height;
        attr.selected_rows_size = grad_rows.size();
        PADDLE_ENFORCE_EQ(attr.grad_width, attr.param_width);

        auto sgd =
            jit::Get<jit::kSgd, jit::SgdTuples<T>, platform::CPUPlace>(attr);
        sgd(lr, param_data, grad_data, rows_data, out_data, &attr);*/
      } else {
        PADDLE_THROW("Unsupported Variable Type of Grad");
      }
    } else if (param_var->IsType<framework::SelectedRows>()) {
      PADDLE_ENFORCE(grad_var->IsType<framework::SelectedRows>(),
                     "when param "
                     "is SelectedRows, gradient should also be SelectedRows");
      const auto &param = param_var->Get<framework::SelectedRows>();
      auto *param_out = ctx.Output<framework::SelectedRows>("ParamOut");
      const auto &grad = grad_var->Get<framework::SelectedRows>();

      // for distributed training, a sparse var may be empty,
      // just skip updating.
      if (grad.rows().size() == 0) {
        return;
      }

      auto param_row_width = param.value().dims()[1];
      auto grad_row_width = grad.value().dims()[1];
      VLOG(4) << " param rows: " << param.rows().size()
              << " param memory rows: " << param.value().dims()[0]
              << " grad rows: " << grad.rows().size()
              << " grad memory rows: " << grad.value().dims()[0];
      PADDLE_ENFORCE_EQ(param_row_width, grad_row_width,
                        "param_row should have the same size with grad_row");

      const auto *lr = learning_rate->data<T>();
      const auto *grad_data = grad.value().data<T>();
      auto *out_data = param_out->mutable_value()->data<T>();
      for (size_t i = 0; i < grad.rows().size(); i++) {
        int64_t id_index = param_out->AutoGrownIndex(grad.rows()[i], false);
        PADDLE_ENFORCE_GE(id_index, static_cast<int64_t>(0),
                          "id should be in the table");
        for (int64_t j = 0; j < grad_row_width; j++) {
          if (188574148 == id_index * grad_row_width + j && grad_row_width == 1) {
            VLOG(1) << "qxz:  " << id_index * grad_row_width + j << " " << out_data[id_index * grad_row_width + j] << " - " << grad_data[i * grad_row_width + j] <<
            " * " << lr[0] ;
          }
          out_data[id_index * grad_row_width + j] -=
              lr[0] * grad_data[i * grad_row_width + j];
        }
        //sse_axpy(&grad_data[i * grad_row_width], &out_data[id_index * grad_row_width], grad_row_width, static_cast<T>(-1*lr[0]));
      }
    } else {
      PADDLE_THROW("Unsupported Variable Type of Parameter");
    }
  }
};
}  // namespace operators
}  // namespace paddle

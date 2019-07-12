/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <cmath>
#include <type_traits>
#include <vector>
#include "paddle/fluid/operators/jit/kernel_base.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace jit {
namespace more {
namespace mkl {

template <typename T>
void MatMul(const T* a, const T* b, T* c, const matmul_attr_t* attr);

template <typename T>
void VMul(const T* x, const T* y, T* z, int n);

template <typename T>
void VAdd(const T* x, const T* y, T* z, int n);

template <typename T>
void VScal(const T* a, const T* x, T* y, int n);

template <typename T>
void VExp(const T* x, T* y, int n);

template <typename T>
void VSquare(const T* x, T* y, int n);

template <typename T>
void VCopy(const T* x, T* y, int n);

template <typename T>
void VAXPY(T a, const T* x, T* y, int n);

template <typename T>
void VBroadcast(const T* x, T* y, int64_t y_h, int64_t x_len) {
  for (int64_t h = 0; h < y_h; ++h) {
    VCopy(x, y + h * x_len, x_len);
  }
}

template <typename T>
void VSigmoid(const T* x, T* y, int n) {
  const T min = SIGMOID_THRESHOLD_MIN;
  const T max = SIGMOID_THRESHOLD_MAX;
  for (int i = 0; i < n; ++i) {
    y[i] = (x[i] < min) ? min : ((x[i] > max) ? max : x[i]);
    y[i] = static_cast<T>(0) - y[i];
  }
  VExp(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(1) / (static_cast<T>(1) + y[i]);
  }
}

template <typename T>
void VTanh(const T* x, T* y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * x[i];
  }
  VSigmoid(y, y, n);
  for (int i = 0; i < n; ++i) {
    y[i] = static_cast<T>(2) * y[i] - static_cast<T>(1);
  }
}

template <typename T>
void SeqPool(const T* x, T* y, const seq_pool_attr_t* attr) {
  VCopy<T>(x, y, attr->w);
  for (int h = 1; h != attr->h; ++h) {
    VAXPY<T>(static_cast<T>(1), x + h * attr->w, y, attr->w);
  }
  if (attr->type == SeqPoolType::kAvg || attr->type == SeqPoolType::kSqrt) {
    T scalar = static_cast<T>(1);
    if (attr->type == SeqPoolType::kAvg) {
      scalar = scalar / static_cast<T>(attr->h);
    } else {
      scalar = scalar / std::sqrt(static_cast<T>(attr->h));
    }
    VScal<T>(&scalar, y, y, attr->w);
  }
}

template <typename T>
void EmbSeqPool(const T* table, const int64_t* idx, T* out,
                const emb_seq_pool_attr_t* attr) {
  PADDLE_ENFORCE_EQ(attr->table_width * attr->index_width, attr->out_width);
  auto check_idx_value_valid = [&](int64_t i) {
    PADDLE_ENFORCE_LT(idx[i], attr->table_height, "idx value: %d, i: %d",
                      idx[i], i);
    PADDLE_ENFORCE_GE(idx[i], 0, "idx value: %d, i: %d", idx[i], i);
  };

  for (int64_t w = 0; w != attr->index_width; ++w) {
    check_idx_value_valid(w);
    VCopy<T>(table + idx[w] * attr->table_width, out + w * attr->table_width,
             attr->table_width);
  }

  for (int64_t h = 1; h < attr->index_height; ++h) {
    for (int64_t w = 0; w < attr->index_width; ++w) {
      int64_t i = h * attr->index_width + w;
      check_idx_value_valid(i);
      VAXPY<T>(static_cast<T>(1), table + idx[i] * attr->table_width,
               out + w * attr->table_width, attr->table_width);
    }
  }
}

template <typename T>
void ASum(const T* x, T* res, int n);

template <typename T>
void StrideASum(const T* x, T* res, int n, int stride);

template <typename T>
void StrideScal(const T* a, const T* x, T* y, int n, int stride);

// remain is the product of dimension shapes after the axis dimension
template <typename T>
void Softmax(const T* x, T* y, int n, int bs, int remain = 1) {
  std::vector<T> entities(bs);
  for (int i = 0; i < bs; ++i) {
    entities[i] = x[i * n];
    for (int c = 1; c < n; ++c) {
      entities[i] = x[i * n + c] > entities[i] ? x[i * n + c] : entities[i];
    }
    for (int c = 0; c < n; ++c) {
      y[i * n + c] = x[i * n + c] - entities[i];
    }
  }
  VExp(y, y, n * bs);
  for (int i = 0; i < bs; ++i) {
    T sum;
    if (remain == 1) {
      ASum(&y[i * n], &sum, n);
      sum = static_cast<T>(1) / sum;
      VScal(&sum, &y[i * n], &y[i * n], n);
    } else {
      for (int j = 0; j < remain; ++j) {
        StrideASum(&y[i * n + j], &sum, n, remain);
        sum = static_cast<T>(1) / sum;
        StrideScal(&sum, &y[i * n + j], &y[i * n + j], n, remain);
      }
    }
  }
}

template <typename T>
void Sgd(const T* lr, const T* param, const T* grad, const int64_t* rows,
         T* out, const sgd_attr_t* attr) {
  PADDLE_ENFORCE_EQ(attr->param_width, attr->grad_width);
  PADDLE_ENFORCE_LE(attr->selected_rows_size, attr->grad_height);
  T scalar = -lr[0];
  int width = attr->grad_width;
  if (out == param) {
    for (int64_t i = 0; i < attr->selected_rows_size; ++i) {
      auto h_idx = rows[i];
      PADDLE_ENFORCE_LT(h_idx, attr->param_height);
      PADDLE_ENFORCE_GE(h_idx, 0);
      VAXPY(scalar, grad + i * width, out + h_idx * width, width);
    }
  } else {
    for (int64_t i = 0; i < attr->selected_rows_size; ++i) {
      auto h_idx = rows[i];
      PADDLE_ENFORCE_LT(h_idx, attr->param_height);
      PADDLE_ENFORCE_GE(h_idx, 0);
      VScal(&scalar, grad + i * width, out + h_idx * width, width);
      VAdd(param + h_idx * width, out + h_idx * width, out + h_idx * width,
           width);
    }
  }
}

#define DECLARE_MKL_KERNEL(name)                                              \
  template <typename T>                                                       \
  class name##Kernel : public KernelMore<name##Tuple<T>> {                    \
   public:                                                                    \
    name##Kernel() { this->func = name<T>; }                                  \
    bool CanBeUsed(const typename name##Tuple<T>::attr_type&) const override; \
    const char* ImplType() const override { return "MKL"; }                   \
  }

// ABCMNK
DECLARE_MKL_KERNEL(MatMul);

// XYZN
DECLARE_MKL_KERNEL(VMul);
DECLARE_MKL_KERNEL(VAdd);

// AXYN
DECLARE_MKL_KERNEL(VScal);
DECLARE_MKL_KERNEL(StrideScal);

// XYN
DECLARE_MKL_KERNEL(VExp);
DECLARE_MKL_KERNEL(VSigmoid);
DECLARE_MKL_KERNEL(VTanh);
DECLARE_MKL_KERNEL(VSquare);
DECLARE_MKL_KERNEL(VCopy);

// others
DECLARE_MKL_KERNEL(SeqPool);
DECLARE_MKL_KERNEL(EmbSeqPool);
DECLARE_MKL_KERNEL(Softmax);
DECLARE_MKL_KERNEL(Sgd);
DECLARE_MKL_KERNEL(VBroadcast);

#undef DECLARE_MKL_KERNEL

}  // namespace mkl
}  // namespace more
}  // namespace jit
}  // namespace operators
}  // namespace paddle

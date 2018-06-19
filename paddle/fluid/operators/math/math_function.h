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
#ifdef PADDLE_WITH_MKLML
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#include <mkl_vml_functions.h>
#endif

#ifdef PADDLE_USE_OPENBLAS
#include <cblas.h>
#ifdef LAPACK_FOUND
#include <lapacke.h>
#endif
#endif

#ifndef LAPACK_FOUND
extern "C" {
#include <cblas.h>  // NOLINT
int LAPACKE_sgetrf(int matrix_layout, int m, int n, float* a, int lda,
                   int* ipiv);
int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a, int lda,
                   int* ipiv);
int LAPACKE_sgetri(int matrix_layout, int n, float* a, int lda,
                   const int* ipiv);
int LAPACKE_dgetri(int matrix_layout, int n, double* a, int lda,
                   const int* ipiv);
}
#endif

#include <cmath>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {
template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context, const framework::Tensor& in,
                  framework::Tensor* out, const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context, framework::Tensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value);

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& vec, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle

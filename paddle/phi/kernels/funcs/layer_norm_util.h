// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
namespace funcs {

// Wrap RowwiseMean and ColwiseMean.
// Reuse the cpu codes and replace the gpu codes with cublas_gemv, which is
// significantly faster. Unlike the RowwiseMean and ColwiseMean, the
// implementation only considers 2D.
template <typename DeviceContext, typename T>
struct RowwiseMean2D {
  RowwiseMean2D(int left, int right, const DeviceContext& dev_ctx);

  void operator()(const DeviceContext& context,
                  const DenseTensor& input,
                  DenseTensor* vec);
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class RowwiseMean2D<phi::GPUContext, T> {
 public:
  RowwiseMean2D(int left, int right, const DeviceContext& dev_ctx)
      : left_(left), right_(right) {
    DDim ones_dim({right_});
    divisor_.Resize(ones_dim);
    dev_ctx.template Alloc<T>(&divisor_);
    phi::funcs::set_constant(dev_ctx, &divisor_, 1.0 / right);
  }
  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  DenseTensor* out) {
    phi::funcs::GetBlas<phi::GPUContext, T>(context).GEMV(false,
                                                          left_,
                                                          right_,
                                                          1.,
                                                          input.data<T>(),
                                                          divisor_.data<T>(),
                                                          0.,
                                                          out->data<T>());
  }

 private:
  int left_;
  int right_;
  DenseTensor divisor_;
};
#endif

template <typename T>
class RowwiseMean2D<phi::CPUContext, T> {
 public:
  RowwiseMean2D(int left UNUSED,
                int right UNUSED,
                const DeviceContext& dev_ctx UNUSED) {}

  void operator()(const phi::CPUContext& context,
                  const DenseTensor& input,
                  DenseTensor* out) {
    row_mean_(context, input, out);
  }

 private:
  phi::funcs::RowwiseMean<phi::CPUContext, T> row_mean_;
};

template <typename DeviceContext, typename T>
struct ColwiseSum2D {
  ColwiseSum2D(int left, int right, const DeviceContext& dev_ctx);

  void operator()(const phi::DeviceContext& context,
                  const DenseTensor& input,
                  DenseTensor* vec);
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
class ColwiseSum2D<phi::GPUContext, T> {
 public:
  ColwiseSum2D(int left, int right, const phi::GPUContext& dev_ctx)
      : left_(left), right_(right) {
    DDim ones_dim({left_});
    divisor_.Resize(ones_dim);
    dev_ctx.template Alloc<T>(&divisor_);
    phi::funcs::set_constant(dev_ctx, &divisor_, 1.0);
  }

  void operator()(const phi::GPUContext& context,
                  const DenseTensor& input,
                  DenseTensor* out) {
    phi::funcs::GetBlas<phi::GPUContext, T>(context).GEMV(true,
                                                          left_,
                                                          right_,
                                                          1.,
                                                          input.data<T>(),
                                                          divisor_.data<T>(),
                                                          0.,
                                                          out->data<T>());
  }

 private:
  int left_;
  int right_;
  DenseTensor divisor_;
};
#endif

template <typename T>
class ColwiseSum2D<phi::CPUContext, T> {
 public:
  ColwiseSum2D(int left UNUSED,
               int right UNUSED,
               const phi::CPUContext& dev_ctx UNUSED) {}

  void operator()(const phi::CPUContext& context,
                  const DenseTensor& input,
                  DenseTensor* out) {
    col_wise_(context, input, out);
  }

 private:
  phi::funcs::ColwiseSum<phi::CPUContext, T> col_wise_;
};

template <typename T>
struct SubAndSquareFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return (a - b) * (a - b); }
};

template <typename T>
struct DivAndSqrtFunctor {
  explicit DivAndSqrtFunctor(T epsilon) { epsilon_ = epsilon; }
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a / (sqrt(b + epsilon_));
  }

 private:
  T epsilon_;
};

template <typename T>
struct MulInvVarFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const {
    return a * std::sqrt(1.0 / b);
  }
};

}  // namespace funcs
}  // namespace phi

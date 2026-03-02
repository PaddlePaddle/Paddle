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

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/sleef_vectorized_math.h"
// #include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

#define ToString(x) #x

template <typename T, typename U, typename Context, typename Functor>
void ActivationImpl(const Context& dev_ctx,
                    const DenseTensor& X,
                    DenseTensor* Out,
                    const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(Out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<U>(Out);
  if (Out->numel() == 0) {
    return;
  }
  auto x = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&X, "Input", "X", "Activation"));
  auto out = phi::EigenVector<U>::Flatten(
      GET_DATA_SAFELY(Out, "Output", "Out", "Activation"));
  auto* place = dev_ctx.eigen_device();
  // use 32bit index to speed up computation
  bool use_32bit_index = out.size() < Eigen::NumTraits<int>::highest();
  bool is_gpu_place = dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU;
  if (use_32bit_index && is_gpu_place) {
    functor(*place, To32BitIndex(x), To32BitIndex(out));
  } else {
    functor(*place, x, out);
  }
}

// Vectorized Sin implementation for CPU - matches PyTorch precision
// Only enabled for float/double on CPU to ensure bit-level alignment
template <typename T, typename Context>
void VectorizedSinImpl(const Context& dev_ctx,
                       const DenseTensor& X,
                       DenseTensor* Out) {
  PADDLE_ENFORCE_NOT_NULL(Out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(Out);
  if (Out->numel() == 0) {
    return;
  }

  const T* x_data = X.data<T>();
  T* out_data = Out->data<T>();
  int64_t numel = X.numel();

  // Check if data is contiguous and use vectorized path
  if (funcs::sleef_vec::should_use_vectorized_path(x_data, out_data, numel)) {
    funcs::sleef_vec::vsin(out_data, x_data, numel);
  } else {
    // Fallback to Eigen-based implementation
    auto x = EigenVector<T>::Flatten(GET_DATA_SAFELY(&X, "Input", "X", "Sin"));
    auto out =
        EigenVector<T>::Flatten(GET_DATA_SAFELY(Out, "Output", "Out", "Sin"));
    auto* place = dev_ctx.eigen_device();
    out.device(*place) = x.unaryExpr(funcs::Sine<T>()).eval();
  }
}

// Vectorized Cos implementation for CPU - matches PyTorch precision
template <typename T, typename Context>
void VectorizedCosImpl(const Context& dev_ctx,
                       const DenseTensor& X,
                       DenseTensor* Out) {
  PADDLE_ENFORCE_NOT_NULL(Out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(Out);
  if (Out->numel() == 0) {
    return;
  }

  const T* x_data = X.data<T>();
  T* out_data = Out->data<T>();
  int64_t numel = X.numel();

  // Check if data is contiguous and use vectorized path
  if (funcs::sleef_vec::should_use_vectorized_path(x_data, out_data, numel)) {
    funcs::sleef_vec::vcos(out_data, x_data, numel);
  } else {
    // Fallback to Eigen-based implementation
    auto x = EigenVector<T>::Flatten(GET_DATA_SAFELY(&X, "Input", "X", "Cos"));
    auto out =
        EigenVector<T>::Flatten(GET_DATA_SAFELY(Out, "Output", "Out", "Cos"));
    auto* place = dev_ctx.eigen_device();
    out.device(*place) = x.unaryExpr(funcs::Cosine<T>()).eval();
  }
}

template <typename T, typename Context>
void LogitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 double eps,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto eigen_in = EigenVector<T>::Flatten(x);
  auto& place = *dev_ctx.eigen_device();
  auto eigen_p = EigenVector<T>::Flatten(*out);

  funcs::LogitFunctor<T> functor;
  functor(place, eigen_in, eigen_out, eigen_p, eps);
}

}  // namespace phi

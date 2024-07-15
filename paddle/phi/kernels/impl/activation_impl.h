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

#include "paddle/common/macros.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/backends/gpu/musa/mudnn_helper.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
namespace phi {
using GPUDNNDataLayout = phi::backends::gpu::DataLayout;

#define ToString(x) #x

template <typename T, typename U, typename Context, typename Functor>
void ActivationImpl(const Context& dev_ctx,
                    const DenseTensor& X,
                    DenseTensor* Out,
                    const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(Out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<U>(Out);
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

template <typename T, typename Context>
void LogitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 float eps,
                 DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto eigen_in = EigenVector<T>::Flatten(x);
  auto& place = *dev_ctx.eigen_device();
  auto eigen_p = EigenVector<T>::Flatten(*out);

  funcs::LogitFunctor<T> functor;
  functor(place, eigen_in, eigen_out, eigen_p, eps);
}

template <typename T, typename Context>
void PowKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const Scalar& factor,
               DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
#ifndef __MUSACC__
  auto x_flatten = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(&x, "Input", "X", "Activation"));
  auto out_flatten = phi::EigenVector<T>::Flatten(
      GET_DATA_SAFELY(out, "Output", "Out", "Activation"));
  auto* place = dev_ctx.eigen_device();
  phi::funcs::PowFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = factor.to<float>();
  functor(*place, x_flatten, out_flatten);
#else
  if (UNLIKELY(x.dtype() == DataType::INT32 || x.dtype() == DataType::INT64 ||
               x.dtype() == DataType::FLOAT64)) {
    auto __summary__ =
        phi::ErrorSummary("pow does not support int/double/int64_t");
    auto __message__ =
        ::paddle::string::Sprintf("", __summary__.error_message());
    __THROW_ERROR_INTERNAL__(
        phi::ErrorSummary(__summary__.code(), std::move(__message__)));
  }
  phi::backends::gpu::ScopedUnaryDescriptor un_desc;
  backends::gpu::ScopedTensorDescriptor x_scoped_desc;
  backends::gpu::ScopedTensorDescriptor out_scoped_desc;
  auto musa_x = x_scoped_desc.descriptor_with_stride<T>(
      x, GPUDNNDataLayout::kNCHW, common::vectorize<int>(x.dims()));
  auto musa_out = out_scoped_desc.descriptor_with_stride<T>(
      *out, GPUDNNDataLayout::kNCHW, common::vectorize<int>(out->dims()));
  un_desc.desc_.SetAlpha(factor.to<double>());
  un_desc.desc_.SetMode(::musa::dnn::Unary::Mode::POW);
  auto handle = dev_ctx.cudnn_handle();
  un_desc.desc_.Run(*handle, musa_out, musa_x);
#endif
}

}  // namespace phi

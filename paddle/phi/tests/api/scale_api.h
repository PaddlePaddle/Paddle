// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/utils/allocator.h"
#include "paddle/phi/api/lib/utils/storage.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace paddle {
namespace experimental {

PADDLE_API Tensor scale_kernel_context(const Tensor& x,
                                       const Scalar& scale,
                                       float bias,
                                       bool bias_after_scale) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "scale API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "scale API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto kernel_context = phi::KernelContext(dev_ctx);

  auto dense_x = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x.get());

  kernel_context.EmplaceBackAttr(phi::Scalar(scale));
  kernel_context.EmplaceBackAttr(bias);
  kernel_context.EmplaceBackAttr(bias_after_scale);

  auto dense_out = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPhiPlace(kernel_backend)),
      phi::DenseTensorMeta());
  phi::MetaTensor meta_out(dense_out.get());
  phi::UnchangedInferMeta(*dense_x, &meta_out);
  kernel_context.EmplaceBackOutput(dense_out.get());

  Tensor out;
  out.set_impl(dense_out);

  kernel(&kernel_context);
  return out;
}

static void ScaleCPU(DataType kernel_dtype,
                     const phi::CPUContext& dev_ctx,
                     const phi::DenseTensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale,
                     phi::DenseTensor* dense_out) {
  switch (kernel_dtype) {
    case phi::DataType::FLOAT64: {
      phi::ScaleKernel<double>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::FLOAT32: {
      phi::ScaleKernel<float>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::BFLOAT16: {
      phi::ScaleKernel<phi::dtype::bfloat16>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT64: {
      phi::ScaleKernel<int64_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT32: {
      phi::ScaleKernel<int32_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT16: {
      phi::ScaleKernel<int16_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT8: {
      phi::ScaleKernel<int8_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::UINT8: {
      phi::ScaleKernel<uint8_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Detected unsupported data type."
          "Only Float64, Float32, BFloat16, Int64, Int32, Int16, Int8, UInt8 "
          "are supported for now."));
      break;
    }
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
static void ScaleGPU(DataType kernel_dtype,
                     const phi::GPUContext& dev_ctx,
                     const phi::DenseTensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale,
                     phi::DenseTensor* dense_out) {
  switch (kernel_dtype) {
    case phi::DataType::FLOAT64: {
      phi::ScaleKernel<double>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::FLOAT32: {
      phi::ScaleKernel<float>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::FLOAT16: {
      phi::ScaleKernel<phi::dtype::float16>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT64: {
      phi::ScaleKernel<int64_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT32: {
      phi::ScaleKernel<int32_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT16: {
      phi::ScaleKernel<int16_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::INT8: {
      phi::ScaleKernel<int8_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case phi::DataType::UINT8: {
      phi::ScaleKernel<uint8_t>(
          dev_ctx, x, phi::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    default: {
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Detected unsupported data type."
          "Only Float64, Float32, Float16, Int64, Int32, Int16, Int8, UInt8 "
          "are "
          "supported for now."));
      break;
    }
  }
}
#endif

Tensor scale_switch_case(const Tensor& x,
                         const Scalar& scale,
                         float bias,
                         bool bias_after_scale) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "scale API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "scale API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<phi::DenseTensor>(x.impl());

  auto dense_out = std::make_shared<phi::DenseTensor>(
      phi::make_intrusive<paddle::experimental::SharedStorage>(
          phi::TransToPhiPlace(kernel_backend)),
      phi::DenseTensorMeta());
  phi::MetaTensor meta_out(dense_out.get());
  phi::UnchangedInferMeta(*dense_x, &meta_out);

  Tensor out;
  out.set_impl(dense_out);

  switch (kernel_backend) {
    case Backend::CPU:
      ScaleCPU(kernel_data_type,
               static_cast<const phi::CPUContext&>(*dev_ctx),
               *dense_x,
               scale,
               bias,
               bias_after_scale,
               dense_out.get());
      break;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case Backend::GPU:
      ScaleGPU(kernel_data_type,
               static_cast<const phi::GPUContext&>(*dev_ctx),
               *dense_x,
               scale,
               bias,
               bias_after_scale,
               dense_out.get());
      break;
#endif
    default:
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Detected unsupported backend."
          "Only CPU and CUDA Backend are supported for now."
          "Please double check if your backend falls into the above two "
          "categories."));
  }

  return out;
}

}  // namespace experimental
}  // namespace paddle

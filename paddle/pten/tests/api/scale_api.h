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

#include "paddle/pten/api/include/tensor.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/allocator.h"
#include "paddle/pten/common/scalar.h"
#include "paddle/pten/common/scalar_array.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/scale_kernel.h"

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
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
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
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "scale API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "scale API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto kernel_context = pten::KernelContext(dev_ctx);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x);

  kernel_context.EmplaceBackAttr(pten::Scalar(scale));
  kernel_context.EmplaceBackAttr(bias);
  kernel_context.EmplaceBackAttr(bias_after_scale);

  auto out_meta = pten::UnchangedInferMeta(dense_x->meta());
  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));
  kernel_context.EmplaceBackOutput(dense_out);

  Tensor out;
  out.set_impl(dense_out);

  kernel(&kernel_context);
  return out;
}

static void ScaleCPU(DataType kernel_dtype,
                     const pten::CPUContext& dev_ctx,
                     const pten::DenseTensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale,
                     pten::DenseTensor* dense_out) {
  switch (kernel_dtype) {
    case pten::DataType::FLOAT64: {
      pten::Scale<double>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::FLOAT32: {
      pten::Scale<float>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::BFLOAT16: {
      pten::Scale<paddle::platform::bfloat16>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT64: {
      pten::Scale<int64_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT32: {
      pten::Scale<int32_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT16: {
      pten::Scale<int16_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT8: {
      pten::Scale<int8_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::UINT8: {
      pten::Scale<uint8_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
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
                     const pten::GPUContext& dev_ctx,
                     const pten::DenseTensor& x,
                     const Scalar& scale,
                     float bias,
                     bool bias_after_scale,
                     pten::DenseTensor* dense_out) {
  switch (kernel_dtype) {
    case pten::DataType::FLOAT64: {
      pten::Scale<double>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::FLOAT32: {
      pten::Scale<float>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::FLOAT16: {
      pten::Scale<paddle::platform::float16>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT64: {
      pten::Scale<int64_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT32: {
      pten::Scale<int32_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT16: {
      pten::Scale<int16_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::INT8: {
      pten::Scale<int8_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
      break;
    }
    case pten::DataType::UINT8: {
      pten::Scale<uint8_t>(
          dev_ctx, x, pten::Scalar(scale), bias, bias_after_scale, dense_out);
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
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
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
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "scale", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "scale API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "scale API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::UnchangedInferMeta(dense_x->meta());
  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  switch (kernel_backend) {
    case Backend::CPU:
      ScaleCPU(kernel_data_type,
               static_cast<const pten::CPUContext&>(*dev_ctx),
               *dense_x,
               scale,
               bias,
               bias_after_scale,
               dense_out.get());
      break;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case Backend::GPU:
      ScaleGPU(kernel_data_type,
               static_cast<const pten::GPUContext&>(*dev_ctx),
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

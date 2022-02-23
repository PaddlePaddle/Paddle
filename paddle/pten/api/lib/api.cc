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

#include "paddle/pten/api/include/api.h"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/kernel_signature.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_declare.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/include/core.h"
#include "paddle/pten/include/infermeta.h"
#include "paddle/pten/kernels/declarations.h"

namespace paddle {
namespace experimental {

PADDLE_API Tensor add(const Tensor& x, const Tensor& y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "add", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "add API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "add API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta =
      pten::ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::add_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, *dense_y, -1, dense_out.get());

  return out;
}

PADDLE_API Tensor cast(const Tensor& x, DataType out_dtype) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

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
      "cast", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "cast API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "cast API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::CastInferMeta(dense_x->meta(), out_dtype);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::cast_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, out_dtype, x.dtype(), dense_out.get());

  return out;
}

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "divide", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "divide API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "divide API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta =
      pten::ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::divide_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, *dense_y, -1, dense_out.get());

  return out;
}

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "dot", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "dot API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "dot API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta = pten::DotInferMeta(dense_x->meta(), dense_y->meta());

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::dot_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, *dense_y, dense_out.get());

  return out;
}

PADDLE_API Tensor empty(const ScalarArray& shape,
                        DataType dtype,
                        Backend place,
                        DataLayout layout) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_layout = ParseLayout(layout);

  kernel_data_type = ParseDataType(dtype);

  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "empty", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "empty API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "empty API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto out_meta = pten::CreateInferMeta(shape, dtype, layout);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::empty_kernel>();
  (*kernel_fn)(*dev_ctx, pten::ScalarArray(shape), dense_out.get());

  return out;
}

PADDLE_API Tensor empty_like(const Tensor& x,
                             DataType dtype,
                             Backend place,
                             DataLayout layout) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackendWithInputOrder(place, x);

  kernel_layout = ParseLayoutWithInputOrder(layout, x);

  kernel_data_type = ParseDataTypeWithInputOrder(dtype, x);

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
      "empty_like", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "empty_like API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "empty_like API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::CreateLikeInferMeta(dense_x->meta(), dtype, layout);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::empty_like_kernel>();
  (*kernel_fn)(*dev_ctx, dense_out.get());

  return out;
}

PADDLE_API Tensor flatten(const Tensor& x, int start_axis, int stop_axis) {
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
      "flatten", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "flatten API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "flatten API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta =
      pten::FlattenInferMeta(dense_x->meta(), start_axis, stop_axis);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::flatten_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, start_axis, stop_axis, dense_out.get());

  return out;
}

PADDLE_API Tensor full(const ScalarArray& shape,
                       const Scalar& value,
                       DataType dtype,
                       Backend place,
                       DataLayout layout) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_layout = ParseLayout(layout);

  kernel_data_type = ParseDataType(dtype);

  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "full", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "full API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "full API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto out_meta = pten::CreateInferMeta(shape, dtype, layout);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::full_kernel>();
  (*kernel_fn)(
      *dev_ctx, pten::ScalarArray(shape), pten::Scalar(value), dense_out.get());

  return out;
}

PADDLE_API Tensor full_like(const Tensor& x,
                            const Scalar& value,
                            DataType dtype,
                            Backend place,
                            DataLayout layout) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackendWithInputOrder(place, x);

  kernel_layout = ParseLayoutWithInputOrder(layout, x);

  kernel_data_type = ParseDataTypeWithInputOrder(dtype, x);

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
      "full_like", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "full_like API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "full_like API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::CreateLikeInferMeta(dense_x->meta(), dtype, layout);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::full_like_kernel>();
  (*kernel_fn)(*dev_ctx, pten::Scalar(value), dense_out.get());

  return out;
}

PADDLE_API Tensor matmul(const Tensor& x,
                         const Tensor& y,
                         bool transpose_x,
                         bool transpose_y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "matmul", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "matmul API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "matmul API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta = pten::MatmulInferMeta(
      dense_x->meta(), dense_y->meta(), transpose_x, transpose_y);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::matmul_kernel>();
  (*kernel_fn)(
      *dev_ctx, *dense_x, *dense_y, transpose_x, transpose_y, dense_out.get());

  return out;
}

PADDLE_API Tensor mean(const Tensor& x,
                       const std::vector<int64_t>& axis,
                       bool keep_dim) {
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
      "mean", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "mean API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "mean API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::ReduceInferMeta(dense_x->meta(), axis, keep_dim);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::mean_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, axis, keep_dim, false, dense_out.get());

  return out;
}

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "multiply", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "multiply API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "multiply API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta =
      pten::ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::multiply_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, *dense_y, -1, dense_out.get());

  return out;
}

PADDLE_API Tensor ones_like(const Tensor& x,
                            DataType dtype,
                            Backend place,
                            DataLayout layout) {
  return full_like(x, 1, dtype, place, layout);
}

PADDLE_API Tensor reshape(const Tensor& x, const ScalarArray& shape) {
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
      "reshape", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "reshape API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "reshape API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::ReshapeInferMeta(dense_x->meta(), shape);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::reshape_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, pten::ScalarArray(shape), dense_out.get());

  return out;
}

PADDLE_API Tensor scale(const Tensor& x,
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

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::scale_kernel>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               pten::Scalar(scale),
               bias,
               bias_after_scale,
               dense_out.get());

  return out;
}

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
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
      "subtract", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "subtract API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "subtract API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  auto dense_y = std::dynamic_pointer_cast<pten::DenseTensor>(y.impl());

  auto out_meta =
      pten::ElementwiseInferMeta(dense_x->meta(), dense_y->meta(), -1);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::subtract_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, *dense_y, -1, dense_out.get());

  return out;
}

PADDLE_API Tensor sum(const Tensor& x,
                      const std::vector<int64_t>& axis,
                      DataType dtype,
                      bool keep_dim) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

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
      "sum", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "sum API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "sum API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::ReduceInferMeta(dense_x->meta(), axis, keep_dim, dtype);

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::sum_kernel>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               axis,
               keep_dim,
               false,
               DataType::UNDEFINED,
               dense_out.get());

  return out;
}

PADDLE_API Tensor zeros_like(const Tensor& x,
                             DataType dtype,
                             Backend place,
                             DataLayout layout) {
  return full_like(x, 0, dtype, place, layout);
}

PADDLE_API Tensor conj(const Tensor& x) {
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
      "conj", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "conj API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "conj API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());

  auto out_meta = pten::UnchangedInferMeta(dense_x->meta());

  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToFluidPlace(kernel_backend)),
      std::move(out_meta));

  Tensor out;
  out.set_impl(dense_out);

  auto* kernel_fn = kernel.GetVariadicKernelFn<pten::conj_kernel>();
  (*kernel_fn)(*dev_ctx, *dense_x, dense_out.get());

  return out;
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Creation);
PT_REGISTER_API(Linalg);
PT_REGISTER_API(Manipulation);
PT_REGISTER_API(Math);

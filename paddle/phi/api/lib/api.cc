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

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/data_transform.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/infermeta/binary.h"
#include "paddle/pten/infermeta/multiary.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"
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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ElementwiseInferMeta(
      MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::CastInferMeta(MakeMetaTensor(*input_x), out_dtype, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    DataType,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, out_dtype, kernel_out);

  return out;
}

PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis) {
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
      "concat", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "concat API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "concat API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ConcatInferMeta(MakeMetaTensor(*input_x), axis, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const std::vector<pten::DenseTensor>&,
                                    const pten::Scalar&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, pten::Scalar(axis), kernel_out);

  return out;
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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ElementwiseInferMeta(
      MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::DotInferMeta(
      MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);

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

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::CreateInferMeta(shape, dtype, layout, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::ScalarArray&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, pten::ScalarArray(shape), kernel_out);

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

  auto input_x = TensorToDenseTensor(x);

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, layout, &meta_out);

  using kernel_signature =
      void (*)(const platform::DeviceContext&, pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::FlattenInferMeta(
      MakeMetaTensor(*input_x), start_axis, stop_axis, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    int,
                                    int,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, kernel_out);

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

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::CreateInferMeta(shape, dtype, layout, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::ScalarArray&,
                                    const pten::Scalar&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(
      *dev_ctx, pten::ScalarArray(shape), pten::Scalar(value), kernel_out);

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

  auto input_x = TensorToDenseTensor(x);

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, layout, &meta_out);

  using kernel_signature = void (*)(
      const platform::DeviceContext&, const pten::Scalar&, pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, pten::Scalar(value), kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::MatmulInferMeta(MakeMetaTensor(*input_x),
                        MakeMetaTensor(*input_y),
                        transpose_x,
                        transpose_y,
                        &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    bool,
                                    bool,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(
      *dev_ctx, *input_x, *input_y, transpose_x, transpose_y, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ReduceInferMeta(MakeMetaTensor(*input_x), axis, keep_dim, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const std::vector<int64_t>&,
                                    bool,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, axis, keep_dim, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ElementwiseInferMeta(
      MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);

  return out;
}

Tensor ones_like(const Tensor& x,
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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ReshapeInferMeta(MakeMetaTensor(*input_x), shape, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::ScalarArray&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, pten::ScalarArray(shape), kernel_out);

  return out;
}

PADDLE_API Tensor scale(const Tensor& x,
                        const Scalar& scale,
                        float bias,
                        bool bias_after_scale) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  KernelType kernel_type = ParseKernelTypeByInputArgs(x);

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

  if (kernel_type == KernelType::DENSE_TENSOR_KENREL) {
    auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale", {kernel_backend, kernel_layout, kernel_data_type});
    VLOG(6) << "scale API kernel key: [" << kernel_backend << ", "
            << kernel_layout << ", " << kernel_data_type << "]";
    VLOG(6) << "scale API kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor out;
    auto kernel_out = SetKernelOutput(kernel_backend, &out);
    pten::MetaTensor meta_out(kernel_out);

    pten::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    using kernel_signature = void (*)(const platform::DeviceContext&,
                                      const pten::DenseTensor&,
                                      const pten::Scalar&,
                                      float,
                                      bool,
                                      pten::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    (*kernel_fn)(*dev_ctx,
                 *input_x,
                 pten::Scalar(scale),
                 bias,
                 bias_after_scale,
                 kernel_out);

    return out;
  } else {
    auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_sr", {kernel_backend, kernel_layout, kernel_data_type});
    VLOG(6) << "scale API SelectedRows kernel key: [" << kernel_backend << ", "
            << kernel_layout << ", " << kernel_data_type << "]";
    VLOG(6) << "scale API SelectedRows kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor out;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &out);
    pten::MetaTensor meta_out(kernel_out);

    pten::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

    using kernel_signature = void (*)(const platform::DeviceContext&,
                                      const pten::SelectedRows&,
                                      const pten::Scalar&,
                                      float,
                                      bool,
                                      pten::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    (*kernel_fn)(*dev_ctx,
                 *input_x,
                 pten::Scalar(scale),
                 bias,
                 bias_after_scale,
                 kernel_out);

    return out;
  }
}

PADDLE_API Tensor sign(const Tensor& x) {
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
      "sign", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "sign API kernel key: [" << kernel_backend << ", " << kernel_layout
          << ", " << kernel_data_type << "]";
  VLOG(6) << "sign API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::ElementwiseInferMeta(
      MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);

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

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor out;
  auto kernel_out = SetKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(kernel_out);

  pten::SumInferMeta(
      MakeMetaTensor(*input_x), axis, dtype, keep_dim, &meta_out);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const std::vector<int64_t>&,
                                    DataType,
                                    bool,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *input_x, axis, dtype, keep_dim, kernel_out);

  return out;
}

Tensor zeros_like(const Tensor& x,
                  DataType dtype,
                  Backend place,
                  DataLayout layout) {
  return full_like(x, 0, dtype, place, layout);
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Math);

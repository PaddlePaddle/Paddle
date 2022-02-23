/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/api/include/strings_api.h"

#include "paddle/phi/api/lib/api_registry.h"
#include "paddle/phi/api/lib/api_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/strings_api_utils.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/string_tensor.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/kernels/declarations.h"

namespace paddle {
namespace experimental {
namespace strings {

PADDLE_API Tensor empty(const ScalarArray& shape, Backend place) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::ALL_LAYOUT;
  DataType kernel_data_type = DataType::STRING;

  kernel_backend = ParseBackend(place);
  // 1. Get kernel signature and kernel
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_empty", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "stings::empty API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "stings::empty API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  phi::MetaTensor meta_out(strings_out);
  phi::CreateInferMeta(shape, kernel_data_type, &meta_out);

  // 4. Run kernel_fn
  using kernel_signature = void (*)(
      const phi::DeviceContext&, const phi::ScalarArray&, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, phi::ScalarArray(shape), strings_out);

  return out;
}

PADDLE_API Tensor empty_like(const Tensor& x, Backend place) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::ALL_LAYOUT;
  DataType kernel_data_type = DataType::STRING;

  kernel_backend = ParseBackendWithInputOrder(place, x);
  if (kernel_backend == Backend::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
    kernel_backend = kernel_key.backend();
  }

  // 1. Get kernel signature and kernel
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_empty_like", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "stings::empty_like API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "stings::empty_like API kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto strings_x = TensorToStringTensor(x);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  phi::MetaTensor meta_out(strings_out);
  phi::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, &meta_out);

  using kernel_signature =
      void (*)(const phi::DeviceContext&, phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, strings_out);

  return out;
}

PADDLE_API Tensor lower(const Tensor& x, const std::string& encoding) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::ALL_LAYOUT;
  DataType kernel_data_type = DataType::STRING;

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  if (kernel_backend == Backend::UNDEFINED) {
    kernel_backend = kernel_key.backend();
  }

  // 1. Get kernel signature and kernel
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_lower", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "strings::lower API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "strings::lower  API kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto strings_x = TensorToStringTensor(x);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  phi::MetaTensor meta_out(strings_out);
  phi::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, &meta_out);

  using kernel_signature = void (*)(const phi::DeviceContext&,
                                    const phi::StringTensor&,
                                    const std::string&,
                                    phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *strings_x, encoding, strings_out);

  return out;
}

PADDLE_API Tensor upper(const Tensor& x, const std::string& encoding) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::ALL_LAYOUT;
  DataType kernel_data_type = DataType::STRING;

  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  if (kernel_backend == Backend::UNDEFINED) {
    kernel_backend = kernel_key.backend();
  }

  // 1. Get kernel signature and kernel
  auto kernel = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_upper", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "strings::upper API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "strings::upper  API kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto strings_x = TensorToStringTensor(x);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  phi::MetaTensor meta_out(strings_out);
  phi::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, &meta_out);

  using kernel_signature = void (*)(const phi::DeviceContext&,
                                    const phi::StringTensor&,
                                    const std::string&,
                                    phi::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *strings_x, encoding, strings_out);

  return out;
}

}  // namespace strings
}  // namespace experimental
}  // namespace paddle

PD_REGISTER_API(StringsApi);

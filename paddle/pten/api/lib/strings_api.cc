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

#include "paddle/pten/api/include/strings_api.h"

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/strings_api_utils.h"
#include "paddle/pten/core/kernel_context.h"
#include "paddle/pten/core/string_tensor.h"
#include "paddle/pten/infermeta/nullary.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/declarations.h"

namespace paddle {
namespace experimental {
namespace strings {

PADDLE_API Tensor empty(const ScalarArray& shape, Backend place) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::ALL_LAYOUT;
  DataType kernel_data_type = DataType::STRING;

  kernel_backend = ParseBackend(place);
  // 1. Get kernel signature and kernel
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_empty", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "stings::empty API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "stings::empty API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(strings_out);
  pten::CreateInferMeta(shape, kernel_data_type, kernel_layout, &meta_out);

  // 4. Run kernel_fn
  using kernel_signature = void (*)(const pten::DeviceContext&,
                                    const pten::ScalarArray&,
                                    pten::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, pten::ScalarArray(shape), strings_out);

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
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
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
  pten::MetaTensor meta_out(strings_out);
  pten::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, kernel_layout, &meta_out);

  using kernel_signature =
      void (*)(const pten::DeviceContext&, pten::StringTensor*);
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
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
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
  pten::MetaTensor meta_out(strings_out);
  pten::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, kernel_layout, &meta_out);

  using kernel_signature = void (*)(const pten::DeviceContext&,
                                    const pten::StringTensor&,
                                    const std::string&,
                                    pten::StringTensor*);
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
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "strings_upper", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "strings::lower API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "strings::lower  API kernel: " << kernel;

  // 2. Get Device Context and input
  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);
  auto strings_x = TensorToStringTensor(x);

  // 3. Set output
  Tensor out;
  auto strings_out = SetStringsKernelOutput(kernel_backend, &out);
  pten::MetaTensor meta_out(strings_out);
  pten::CreateLikeInferMeta(
      MakeMetaTensor(*strings_x), kernel_data_type, kernel_layout, &meta_out);

  using kernel_signature = void (*)(const pten::DeviceContext&,
                                    const pten::StringTensor&,
                                    const std::string&,
                                    pten::StringTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx, *strings_x, encoding, strings_out);

  return out;
}

}  // namespace strings
}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(StringsApi);

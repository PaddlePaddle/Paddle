/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/include/manual_api.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/data_transform.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/core/meta_tensor.h"
#include "paddle/pten/infermeta/unary.h"

PT_DECLARE_KERNEL(copy, CPU, ALL_LAYOUT);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PT_DECLARE_KERNEL(copy, GPU, ALL_LAYOUT);
#endif

#ifdef PADDLE_WITH_XPU
PT_DECLARE_KERNEL(copy, XPU, ALL_LAYOUT);
#endif

namespace paddle {
namespace experimental {

PADDLE_API Tensor copy_to(const Tensor& x, Backend backend, bool blocking) {
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      "copy", kernel_key);

  VLOG(0) << "to API kernel key: " << kernel_key;
  VLOG(0) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto dense_x = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  kernel_context.EmplaceBackInput(dense_x.get());
  kernel_context.EmplaceBackAttr(blocking);

  // 4. Prepare outputs & InferMeta
  auto dense_out = std::make_shared<pten::DenseTensor>(
      pten::make_intrusive<paddle::experimental::SharedStorage>(
          pten::TransToPtenPlace(backend)),
      pten::DenseTensorMeta());
  pten::MetaTensor meta_out(dense_out.get());
  pten::UnchangedInferMeta(*dense_x, &meta_out);
  dense_out->mutable_data(pten::TransToPtenPlace(backend));
  kernel_context.EmplaceBackOutput(dense_out.get());
  Tensor out;
  out.set_impl(dense_out);

  // 5. Call kernel
  kernel(&kernel_context);

  return out;
}

PADDLE_API std::vector<Tensor> split(const Tensor& x,
                                     const ScalarArray& num_or_sections,
                                     const Scalar& axis) {
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
      "split", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "split API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "split API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto dense_x = PrepareData(x, kernel.InputAt(0), {});

  // Calculate the number of out tensors
  size_t out_number;
  if (num_or_sections.GetData().size() == 1) {
    out_number = num_or_sections.GetData()[0];
  } else {
    out_number = num_or_sections.GetData().size();
  }

  std::vector<Tensor> out;
  auto dense_outs = SetKernelOutput(out_number, kernel_backend, &out);
  std::vector<pten::MetaTensor> meta_outs;
  for (size_t i = 0; i < out_number; ++i) {
    meta_outs.push_back(dense_outs[i]);
  }

  pten::SplitInferMeta(
      MakeMetaTensor(*dense_x), num_or_sections, axis, &meta_outs);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::ScalarArray&,
                                    const pten::Scalar&,
                                    std::vector<pten::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx,
               *dense_x,
               pten::ScalarArray(num_or_sections),
               pten::Scalar(axis),
               dense_outs);

  return out;
}
}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(Utils);

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

#include "paddle/pten/api/include/sparse_utils.h"

#include <memory>
#include "glog/logging.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
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

PADDLE_API Tensor to_sparse_coo(const Tensor& x,
                                Backend backend,
                                const int64_t sparse_dim) {
  if (x.layout() == pten::DataLayout::SPARSE_COO) {
    return x;
  }
  // 1. Get kernel signature and kernel
  auto kernel_key_set = ParseKernelKeyByInputArgs(x);
  kernel_key_set.backend_set = kernel_key_set.backend_set | BackendSet(backend);
  auto kernel_key = kernel_key_set.GetHigestPriorityKernelKey();
  std::string kernel_name = "dense_to_sparse_coo";
  if (x.layout() == pten::DataLayout::SPARSE_CSR) {
    kernel_name = "sparse_csr_to_coo";
  }

  auto kernel = pten::KernelFactory::Instance().SelectKernelOrThrowError(
      kernel_name, kernel_key);

  VLOG(0) << "to API kernel key: " << kernel_key;
  VLOG(0) << "to API kernel: " << kernel;

  // 2. Get Device Context
  auto* dev_ctx = GetDeviceContextByBackend(kernel_key.backend());
  auto kernel_context = pten::KernelContext(dev_ctx);

  // 3. Auto data transform
  auto input = std::dynamic_pointer_cast<pten::DenseTensor>(x.impl());
  if (x.layout() == pten::DataLayout::SPARSE_CSR) {
    input = std::dynamic_pointer_cast<pten::SparseCsrTensor>(x.impl());
  }

  kernel_context.EmplaceBackInput(input.get());
  kernel_context.EmplaceBackAttr(sparse_dim);

  // 4. InferMeta
  // auto out_meta = UnchangedInferMeta(dense_x->meta());

  // 5. Prepare outputs
  // auto coo = std::make_shared<pten::SparseCooTensor>(
  //    pten::make_intrusive<paddle::experimental::SharedStorage>(
  //        pten::TransToFluidPlace(backend)),
  //    std::move(out_meta));
  // coo->mutable_data(pten::TransToFluidPlace(backend));
  // kernel_context.EmplaceBackOutput(coo.get());
  // Tensor out;
  // out.set_impl(coo);

  // 6. Call kernel
  kernel(&kernel_context);

  return out;
}

}  // namespace experimental
}  // namespace paddle

PT_REGISTER_API(SparseUtils);

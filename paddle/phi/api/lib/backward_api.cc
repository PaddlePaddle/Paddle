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

#include "paddle/pten/api/backward/backward_api.h"
#include <memory>

#include "glog/logging.h"

#include "paddle/pten/api/include/api.h"
#include "paddle/pten/api/lib/api_registry.h"
#include "paddle/pten/api/lib/api_utils.h"
#include "paddle/pten/api/lib/data_transform.h"
#include "paddle/pten/api/lib/kernel_dispatch.h"
#include "paddle/pten/api/lib/utils/storage.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/infermeta/backward.h"

namespace paddle {
namespace experimental {

PADDLE_API std::vector<std::vector<Tensor>> matmul_grad(const Tensor& x,
                                                        const Tensor& y,
                                                        const Tensor& out_grad,
                                                        bool transpose_x,
                                                        bool transpose_y) {
  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED ||
      kernel_layout == DataLayout::UNDEFINED ||
      kernel_data_type == DataType::UNDEFINED) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
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
      "matmul_grad", {kernel_backend, kernel_layout, kernel_data_type});
  VLOG(6) << "matmul_grad API kernel key: [" << kernel_backend << ", "
          << kernel_layout << ", " << kernel_data_type << "]";
  VLOG(6) << "matmul_grad API kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});
  auto input_out_grad = PrepareData(out_grad, kernel.InputAt(2), {});

  std::vector<std::vector<Tensor>> out(2);
  out[0].emplace_back();
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &out[0][0]);
  out[1].emplace_back();
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &out[1][0]);
  pten::MetaTensor meta_out_0(kernel_out_0);
  pten::MetaTensor meta_out_1(kernel_out_1);

  pten::GeneralBinaryGradInferMeta(MakeMetaTensor(*input_x),
                                   MakeMetaTensor(*input_y),
                                   &meta_out_0,
                                   &meta_out_1);

  using kernel_signature = void (*)(const platform::DeviceContext&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    const pten::DenseTensor&,
                                    bool,
                                    bool,
                                    pten::DenseTensor*,
                                    pten::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  (*kernel_fn)(*dev_ctx,
               *input_x,
               *input_y,
               *input_out_grad,
               transpose_x,
               transpose_y,
               kernel_out_0,
               kernel_out_1);

  return out;
}

Tensor scale_grad(const Tensor& out_grad,
                  const Scalar& scale_val,
                  float bias,
                  bool bias_after_scale) {
  return scale(out_grad, scale_val, bias, bias_after_scale);
}

}  // namespace experimental
}  // namespace paddle

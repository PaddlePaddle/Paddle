// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/common/enforce.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void ApFacadeKernel(const Context& dev_ctx,
                    const paddle::optional<std::vector<const DenseTensor*>>& xs,
                    int64_t num_outputs,
                    const std::string& custom_op_name,
                    const std::string& infer_meta_func_name,
                    const std::string& infer_symbolic_func_name,
                    const std::string& serialized_attributes,
                    std::vector<DenseTensor*> outs) {
  PADDLE_THROW(
      common::errors::Unimplemented("ap_facade has no kernel registered."));
}

}  // namespace phi

PD_REGISTER_KERNEL(ap_facade,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApFacadeKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

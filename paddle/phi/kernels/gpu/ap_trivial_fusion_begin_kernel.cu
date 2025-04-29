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

#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "jitify.hpp"  // NOLINT
#include "paddle/common/enforce.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_device_function.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

namespace phi {

template <typename T, typename Context>
void ApTrivialFusionBeginKernel(
    const Context& dev_ctx,
    const paddle::optional<std::vector<const DenseTensor*>>& xs,
    DenseTensor* out) {
  PADDLE_THROW(common::errors::Unimplemented(
      "[DEAD CODE] pd_op.ap_trivial_fusion_begin has no kernel registered."));
}

}  // namespace phi

PD_REGISTER_KERNEL(ap_trivial_fusion_begin,
                   GPU,
                   ALL_LAYOUT,
                   phi::ApTrivialFusionBeginKernel,
                   float,
                   double,
                   int,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

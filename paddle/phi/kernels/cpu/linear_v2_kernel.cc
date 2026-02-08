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
#include <algorithm>
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

#include "paddle/common/enforce.h"
#include "paddle/phi/kernels/addmm_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/impl/matmul_kernel_impl.h"
#include "paddle/phi/kernels/linear_v2_kernel.h"
#include "paddle/phi/kernels/reshape_kernel.h"
#include "paddle/phi/kernels/tile_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/scope_guard.h"

namespace phi {

template <typename T, typename Context>
void LinearV2Kernel(const Context& dev_ctx,
                    const DenseTensor& input,
                    const DenseTensor& weight,
                    const DenseTensor& bias,
                    const bool transpose_weight,
                    DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) {
    return;
  }

  // When in CPU, we use legacy linear_logic by default.
  // TODO(Pan Zhaowu): Adding more efficient kernel for CPU.
  std::vector<std::int64_t> input_dims_vec = common::vectorize(input.dims());
  std::vector<std::int64_t> weight_dims_vec = common::vectorize(weight.dims());

  MatMulFunction<Context, T>(dev_ctx,
                             input,
                             weight,
                             input_dims_vec,
                             weight_dims_vec,
                             out,
                             false,
                             transpose_weight);
  AddKernel<T, Context>(dev_ctx, *out, bias, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    linear_v2, CPU, ALL_LAYOUT, phi::LinearV2Kernel, float, double) {}

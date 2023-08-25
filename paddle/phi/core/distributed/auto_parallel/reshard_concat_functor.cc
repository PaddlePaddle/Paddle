// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/distributed/auto_parallel/reshard_concat_functor.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/concat_kernel.h"

namespace phi {
namespace distributed {

DenseTensor ReshardConcatFunctor(const DeviceContext& dev_ctx,
                                 const std::vector<const DenseTensor*>& input,
                                 int64_t axis) {
  DenseTensor result;
  auto dtype = (*input.begin())->dtype();

  if (phi::CPUContext::classof(&dev_ctx)) {
    PD_VISIT_ALL_TYPES(
        dtype, "Concat", ([&] {
          Concat<data_t>(
              static_cast<const CPUContext&>(dev_ctx), input, axis, &result);
        }));
    return result;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::GPUContext::classof(&dev_ctx)) {
    PD_VISIT_ALL_TYPES(
        dtype, "Concat", ([&] {
          Concat<data_t>(
              static_cast<const GPUContext&>(dev_ctx), input, axis, &result);
        }));
    return result;
  }
#endif
  PADDLE_THROW(phi::errors::Unimplemented(
      "The concat in reshard only supported on CPU and GPU for now."));
}

}  // namespace distributed
}  // namespace phi

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

#include "paddle/phi/core/distributed/auto_parallel/reshard_split_functor.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/visit_type.h"
#include "paddle/phi/kernels/split_kernel.h"

namespace phi {
namespace distributed {

std::vector<DenseTensor> ReshardSplitFunctor(const DeviceContext& dev_ctx,
                                             const DenseTensor& input,
                                             const IntArray& sections,
                                             int64_t axis) {
  size_t out_number = sections.size();
  std::vector<DenseTensor> result(out_number);

  std::vector<MetaTensor> out_meta;
  std::vector<MetaTensor*> out_meta_ptr;

  out_meta.reserve(out_number);
  out_meta_ptr.reserve(out_number);
  for (size_t i = 0; i < out_number; ++i) {
    out_meta.emplace_back(result[i]);
    out_meta_ptr.emplace_back(&out_meta.back());
  }
  SplitInferMeta(phi::MetaTensor(input), sections, axis, out_meta_ptr);

  std::vector<DenseTensor*> outs;
  for (size_t i = 0; i < out_number; ++i) {
    outs.emplace_back(&result[i]);
  }

  if (phi::CPUContext::classof(&dev_ctx)) {
    PD_VISIT_ALL_TYPES(input.dtype(), "SplitKernel", ([&] {
                         SplitKernel<data_t>(
                             static_cast<const CPUContext&>(dev_ctx),
                             input,
                             sections,
                             axis,
                             outs);
                       }));
    return result;
  }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::GPUContext::classof(&dev_ctx)) {
    PD_VISIT_ALL_TYPES(input.dtype(), "SplitKernel", ([&] {
                         SplitKernel<data_t>(
                             static_cast<const GPUContext&>(dev_ctx),
                             input,
                             sections,
                             axis,
                             outs);
                       }));
    return result;
  }
#endif
  PADDLE_THROW(phi::errors::Unimplemented(
      "The split in reshard only supported on CPU and GPU for now."));
}

}  // namespace distributed
}  // namespace phi

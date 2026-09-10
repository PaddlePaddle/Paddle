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

#include "paddle/phi/kernels/unbind_kernel.h"
#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void UnbindStridedKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         int axis,
                         std::vector<DenseTensor*> outs) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  int64_t num = static_cast<int64_t>(outs.size());
  int64_t start = 0;

  axis = axis < 0 ? axis + x.dims().size() : axis;

  for (int64_t i = 0; i < num; i++) {
    SliceStridedKernel<Context>(dev_ctx,
                                x,
                                {axis},
                                IntArray({start}),
                                IntArray({start + 1}),
                                std::vector<int64_t>(),
                                {axis},
                                outs[i]);
    start += 1;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(unbind,
                                         STRIDED,
                                         phi::UnbindStridedKernel) {}

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

#include "paddle/phi/kernels/split_kernel.h"

#include "glog/logging.h"

#include "paddle/common/flags.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(use_stride_kernel);

namespace phi {

template <typename Context>
void SplitStridedKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const IntArray& sections UNUSED,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  int64_t num = static_cast<int64_t>(outs.size());
  int64_t start = 0;

  int axis = axis_scalar.to<int>();

  for (int64_t i = 0; i < num; i++) {
    auto size = outs[i]->dims()[axis];
    SliceStridedKernel<Context>(dev_ctx,
                                x,
                                {axis},
                                IntArray({start}),
                                IntArray({start + size}),
                                std::vector<int64_t>(),
                                std::vector<int64_t>(),
                                outs[i]);
    start += size;
  }
}

template <typename Context>
void SplitWithNumStridedKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               int num,
                               const Scalar& axis_scalar,
                               std::vector<DenseTensor*> outs) {
  if (!FLAGS_use_stride_kernel) {
    PADDLE_THROW(common::errors::Fatal(
        "FLAGS_use_stride_kernel is closed. Strided kernel "
        "be called, something wrong has happened!"));
  }
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  sections_vec.reserve(num);
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  IntArray sections(sections_vec);
  SplitStridedKernel<Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(split_strided,
                                         STRIDED,
                                         phi::SplitStridedKernel) {}

PD_REGISTER_KERNEL_FOR_ALL_BACKEND_DTYPE(split_with_num_strided,
                                         STRIDED,
                                         phi::SplitWithNumStridedKernel) {}

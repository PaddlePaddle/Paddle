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

#include "paddle/phi/kernels/c_split_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void CSplitKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  int rank,
                  int nranks,
                  bool use_model_parallel,
                  DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_GE(rank,
                    0,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_split must be "
                        "greater than or equal to 0.",
                        rank));
  PADDLE_ENFORCE_GE(nranks,
                    2,
                    common::errors::PreconditionNotMet(
                        "The value of nranks (%d) for c_split must be "
                        "greater than or equal to 2.",
                        nranks));
  PADDLE_ENFORCE_LT(rank,
                    nranks,
                    common::errors::PreconditionNotMet(
                        "The value of rank (%d) for c_split must be "
                        "less than that of nranks (%d).",
                        rank,
                        nranks));

  auto dims = x.dims();
  auto dims_size = dims.size();
  // final dim
  int64_t end_size = dims[dims_size - 1];

  // remain dim
  auto remain_ddim = common::slice_ddim(dims, 0, dims_size - 1);
  int64_t remain_numel = common::product(remain_ddim);

  dims[dims_size - 1] /= nranks;
  out->Resize(dims);
  dev_ctx.template Alloc(out, x.dtype());

  std::vector<XPUType*> output_list(nranks, nullptr);
  output_list.at(rank) = reinterpret_cast<XPUType*>(out->data<T>());
  std::vector<int64_t> split_list(nranks, dims[dims_size - 1]);
  int axis = 1;

  auto ret = xpu::split(dev_ctx.x_context(),
                        reinterpret_cast<const XPUType*>(x.data<T>()),
                        output_list,
                        {remain_numel, end_size},
                        split_list,
                        axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(ret, "split");
}
}  // namespace phi

PD_REGISTER_KERNEL(c_split,
                   XPU,
                   ALL_LAYOUT,
                   phi::CSplitKernel,
                   float,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

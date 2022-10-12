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

#include "paddle/phi/kernels/split_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SplitKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const IntArray& sections,
                 const Scalar& axis_scalar,
                 std::vector<DenseTensor*> outs) {
  int axis = axis_scalar.to<int>();
  auto in_dims = x.dims();
  auto input_shape = vectorize<int>(in_dims);
  std::vector<T*> out_ptrs;
  std::vector<int> split_lists;
  for (size_t j = 0; j < outs.size(); ++j) {
    dev_ctx.template Alloc<T>(outs[j]);
    out_ptrs.push_back(outs[j]->data<T>());
    split_lists.push_back(outs[j]->dims()[axis]);
  }
  int r = xpu::split<T>(dev_ctx.x_context(),
                        x.data<T>(),
                        out_ptrs,
                        input_shape,
                        split_lists,
                        axis);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "split");
}

template <typename T, typename Context>
void SplitWithNumKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int num,
                        const Scalar& axis_scalar,
                        std::vector<DenseTensor*> outs) {
  int axis_value = axis_scalar.to<int>();
  auto input_axis_dim = x.dims().at(axis_value);
  std::vector<int64_t> sections_vec;
  for (int i = 0; i < num; ++i) {
    sections_vec.push_back(input_axis_dim / num);
  }
  IntArray sections(sections_vec);
  SplitKernel<T, Context>(dev_ctx, x, sections, axis_scalar, outs);
}

}  // namespace phi

PD_REGISTER_KERNEL(split, XPU, ALL_LAYOUT, phi::SplitKernel, float, int) {}
PD_REGISTER_KERNEL(
    split_with_num, XPU, ALL_LAYOUT, phi::SplitWithNumKernel, float, int) {}

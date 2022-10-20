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

#include "paddle/phi/kernels/roll_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void RollKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& shifts,
                const std::vector<int64_t>& axis,
                DenseTensor* out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  auto shifts_data = shifts.GetData();
  dev_ctx.template Alloc<T>(out);
  DDim input_dim = x.dims();
  std::vector<int> xshape;
  size_t nums = shifts_data.size();
  for (int i = 0; i < input_dim.size(); ++i) {
    xshape.emplace_back(input_dim[i]);
  }

  auto dims = axis;

  // axis = none, reshape to 1-D tensor
  if (dims.size() == 0) {
    dims.push_back(0l);
    input_dim = phi::Dim<1>(x.numel());
  }

  std::vector<int> shifts_in;
  std::vector<int> axis_in;

  for (size_t i = 0; i < nums; ++i) {
    int a = dims[i];
    if (a < 0) {
      a += (input_dim.size());
    }
    axis_in.emplace_back(a);
    int sh = shifts_data[i] % input_dim[a];
    if (sh < 0) {
      sh += input_dim[a];
    }
    shifts_in.emplace_back(sh);
  }

  int r = xpu::roll(dev_ctx.x_context(),
                    reinterpret_cast<const XPUType*>(x.data<T>()),
                    reinterpret_cast<XPUType*>(out->data<T>()),
                    xshape,
                    shifts_in,
                    axis_in);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "roll");
}

}  // namespace phi

PD_REGISTER_KERNEL(roll, XPU, ALL_LAYOUT, phi::RollKernel, float) {}

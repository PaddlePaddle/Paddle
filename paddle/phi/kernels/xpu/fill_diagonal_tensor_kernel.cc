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

#include "paddle/phi/kernels/fill_diagonal_tensor_kernel.h"

#include <array>

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context &dev_ctx,
                              const DenseTensor &x,
                              const DenseTensor &y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              DenseTensor *out) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  T *out_data = dev_ctx.template Alloc<T>(out);
  int r = xpu::copy(dev_ctx.x_context(),
                    reinterpret_cast<const XPUType *>(x.data<T>()),
                    reinterpret_cast<XPUType *>(out_data),
                    x.numel());
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");

  // Compute diagonal positions on CPU (XDNN fill_diagonal_tensor does not
  // support all dtypes that Paddle registers, so we implement the fill logic
  // ourselves using a CPU round-trip for the diagonal data, matching the
  // CPU kernel algorithm).
  auto out_dims = out->dims();
  const auto &matdims = y.dims();
  auto fill_dims = common::flatten_to_2d(matdims, matdims.size() - 1);

  std::array<int64_t, 2> new_dims = {};
  std::array<int64_t, 2> strides = {};
  std::vector<int64_t> matdim;
  matdim.resize(fill_dims[0]);
  CalMatDims(out_dims,
             dim1,
             dim2,
             &offset,
             new_dims.data(),
             strides.data(),
             matdim.data());

  auto place = dev_ctx.GetPlace();
  auto cpu_place = CPUPlace();
  auto size = out->numel();

  // Copy out to CPU, fill diagonal values, then copy back to XPU
  DenseTensor out_cpu;
  out_cpu.Resize(out->dims());
  T *out_cpu_data = dev_ctx.template HostAlloc<T>(&out_cpu);
  memory_utils::Copy(
      cpu_place, out_cpu_data, place, out_data, sizeof(T) * size);

  DenseTensor y_cpu;
  y_cpu.Resize(y.dims());
  T *y_cpu_data = dev_ctx.template HostAlloc<T>(&y_cpu);
  memory_utils::Copy(
      cpu_place, y_cpu_data, place, y.data<T>(), sizeof(T) * y.numel());

  const T *fill_data = y_cpu_data;
  for (int64_t i = 0; i < fill_dims[0]; ++i) {
    auto sumoff = matdim[i] + offset;
    for (int64_t j = 0; j < fill_dims[1]; ++j) {
      auto fill_index = j * (strides[1] + strides[0]) + sumoff;
      if (fill_index < size) {
        out_cpu_data[fill_index] = fill_data[i * fill_dims[1] + j];
      }
    }
  }

  memory_utils::Copy(
      place, out_data, cpu_place, out_cpu_data, sizeof(T) * size);
}
}  // namespace phi

PD_REGISTER_KERNEL(fill_diagonal_tensor,
                   XPU,
                   ALL_LAYOUT,
                   phi::FillDiagonalTensorKernel,
                   float,
                   int64_t,
                   int,
                   phi::float16,
                   bool) {}

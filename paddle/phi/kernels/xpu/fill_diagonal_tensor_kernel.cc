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

#include <algorithm>
#include <vector>

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
  // Copy input to output first (same as CPU/GPU kernels)
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  T *out_data = dev_ctx.template Alloc<T>(out);

  auto out_dims = out->dims();
  const auto &matdims = y.dims();
  auto fill_dims = common::flatten_to_2d(matdims, matdims.size() - 1);

  // Use CalMatDims (same as CPU/GPU kernels) to compute strides,
  // transformed offset, and matoffset array
  std::array<int64_t, 2> new_dims = {};
  std::array<int64_t, 2> strides = {};
  std::vector<int64_t> matdim;
  matdim.resize(fill_dims[0]);
  int64_t computed_offset = offset;
  CalMatDims(out_dims,
             dim1,
             dim2,
             &computed_offset,
             new_dims.data(),
             strides.data(),
             matdim.data());

  PADDLE_ENFORCE_EQ(
      new_dims[0],
      fill_dims[0],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));
  PADDLE_ENFORCE_EQ(
      new_dims[1],
      fill_dims[1],
      errors::InvalidArgument("The dims should be %d x %d, but get "
                              "%d x %d in fill tensor Y",
                              new_dims[0],
                              new_dims[1],
                              fill_dims[0],
                              fill_dims[1]));

  // Compute diagonal fill positions on CPU (same logic as CPU/GPU kernels)
  auto size = out->numel();
  int64_t fill_count = fill_dims[0] * fill_dims[1];

  // Copy fill_data from XPU to CPU first, so we can read it on the host
  // side to compute values_host. The y tensor lives on XPU, so direct
  // host access would segfault. Use synchronous Copy for XPU-to-CPU.
  std::unique_ptr<T[]> fill_data_cpu(new T[y.numel()]);
  memory_utils::Copy(CPUPlace(),
                     reinterpret_cast<void *>(fill_data_cpu.get()),
                     XPUPlace(dev_ctx.GetPlace().GetDeviceId()),
                     reinterpret_cast<const void *>(y.data<T>()),
                     y.numel() * sizeof(T));

  // Use unique_ptr instead of vector to avoid std::vector<bool> specialization
  // which lacks .data() and stores bits instead of contiguous bool objects
  std::vector<int64_t> indices_host(fill_count);
  std::unique_ptr<XPUType[]> values_host(new XPUType[fill_count]);
  int64_t valid_count = 0;

  for (int64_t i = 0; i < fill_dims[0]; i++) {
    auto sumoff = matdim[i] + computed_offset;
    for (int64_t j = 0; j < fill_dims[1]; j++) {
      auto fill_index = j * (strides[1] + strides[0]) + sumoff;
      if (fill_index < size) {
        indices_host[valid_count] = fill_index;
        values_host[valid_count] = reinterpret_cast<const XPUType *>(
            fill_data_cpu.get())[i * fill_dims[1] + j];
        valid_count++;
      }
    }
  }

  if (valid_count == 0) return;

  // Copy index data from CPU to XPU using memory_utils::Copy (CPU-to-device)
  DenseTensor idx_tensor;
  idx_tensor.Resize({static_cast<int64_t>(valid_count)});
  int64_t *idx_data = dev_ctx.template Alloc<int64_t>(&idx_tensor);
  memory_utils::Copy(XPUPlace(dev_ctx.GetPlace().GetDeviceId()),
                     reinterpret_cast<void *>(idx_data),
                     CPUPlace(),
                     reinterpret_cast<const void *>(indices_host.data()),
                     sizeof(int64_t) * valid_count);

  // Copy value data from CPU to XPU using memory_utils::Copy (CPU-to-device)
  DenseTensor val_tensor;
  val_tensor.Resize({static_cast<int64_t>(valid_count)});
  T *val_data_raw = dev_ctx.template Alloc<T>(&val_tensor);
  XPUType *val_data = reinterpret_cast<XPUType *>(val_data_raw);
  memory_utils::Copy(XPUPlace(dev_ctx.GetPlace().GetDeviceId()),
                     reinterpret_cast<void *>(val_data),
                     CPUPlace(),
                     reinterpret_cast<const void *>(values_host.get()),
                     sizeof(XPUType) * valid_count);

  // Use xpu::scatter_element to write values at computed diagonal positions
  // Treat the output as a 1D tensor and scatter at axis=0
  int r = xpu::scatter_element<XPUType, int64_t>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType *>(out_data),
      val_data,
      idx_data,
      reinterpret_cast<XPUType *>(out_data),
      {size},
      {static_cast<int64_t>(valid_count)},
      {static_cast<int64_t>(valid_count)},
      0,
      0);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_element");
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

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

#include "paddle/phi/kernels/tensor_array_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void CreateArrayKernel(const Context& dev_ctx,
                       DataType dtype,
                       TensorArray* out) {}

template <typename T, typename Context>
void ArrayLengthKernel(const Context& dev_ctx,
                       const TensorArray& x,
                       DenseTensor* out) {
  out->Resize({1});
  dev_ctx.template Alloc<int64_t>(out);
  *out->data<int64_t>() = static_cast<int64_t>(x.size());
}

size_t GetOffset(const DenseTensor& i, const phi::DeviceContext& dev_ctx) {
  PADDLE_ENFORCE_EQ(
      i.numel(),
      1,
      errors::InvalidArgument("Input(I) must have numel 1. "
                              "But received %d, and it's shape is [%s].",
                              i.numel(),
                              i.dims()));
  size_t offset;
  if (i.place() == phi::AllocationType::GPU ||
      i.place() == phi::AllocationType::XPU ||
      i.place() == phi::AllocationType::CUSTOM) {
    // FIXME: Avoid copy from GPU to CPU
    phi::DenseTensor t;
    phi::Copy(dev_ctx, i, phi::CPUPlace(), false, &t);
    dev_ctx.Wait();
    offset = static_cast<size_t>(*t.data<int64_t>());
  } else {
    offset = static_cast<size_t>(*i.data<int64_t>());
  }
  return offset;
}

template <typename T, typename Context>
void ArrayWriteKernel(const Context& dev_ctx,
                      const TensorArray& array,
                      const DenseTensor& x,
                      const DenseTensor& i,
                      TensorArray* out) {
  size_t offset = GetOffset(i, dev_ctx);
  if (offset >= out->size()) {
    out->resize(offset + 1);
  }
  auto* out_tensor = &out->at(offset);
  out_tensor->set_lod(x.lod());
  if (x.memory_size() > 0) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out_tensor);
  } else {
    VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                "nothing has been written to output array["
             << offset << "].";
  }
}

}  // namespace phi
PD_REGISTER_KERNEL(create_array,
                   CPU,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   float,
                   double,
                   bool) {}

PD_REGISTER_KERNEL(array_length,
                   CPU,
                   ALL_LAYOUT,
                   phi::ArrayLengthKernel,
                   float,
                   double,
                   bool) {}

PD_REGISTER_KERNEL(
    array_write, CPU, ALL_LAYOUT, phi::ArrayWriteKernel, float, double, bool) {}

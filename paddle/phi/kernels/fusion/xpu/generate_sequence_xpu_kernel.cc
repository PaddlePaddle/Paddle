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

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
namespace fusion {

template <typename T, typename Context>
void GenerateSequenceXPU(const Context& dev_ctx,
                         const DenseTensor& x,
                         DataType dtype,
                         DenseTensor* out) {
  auto x_dims = x.dims();
  int batch = x_dims[0];
  int step = x_dims[1];

  DenseTensor out_host;
  out_host.Resize(x_dims);
  out_host.set_type(dtype);
  T* out_host_data = dev_ctx.template HostAlloc<T>(&out_host);
  for (int i = 0; i < step; i++) {
    out_host_data[i] = static_cast<T>(i);
  }
  for (int i = 1; i < batch; i++) {
    std::memcpy(out_host_data + i * step, out_host_data, step * sizeof(T));
  }

  dev_ctx.template Alloc<T>(out);
  phi::Copy(dev_ctx, out_host, out->place(), true, out);
}

}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(generate_sequence_xpu,
                   XPU,
                   ALL_LAYOUT,
                   phi::fusion::GenerateSequenceXPU,
                   float,
                   int,
                   int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

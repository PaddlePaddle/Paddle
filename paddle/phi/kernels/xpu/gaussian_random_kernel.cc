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

#include "paddle/phi/kernels/gaussian_random_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"

namespace phi {

template <typename T, typename Context>
void GaussianRandomKernel(const Context& ctx,
                          const IntArray& shape,
                          float mean,
                          float std,
                          int seed,
                          DataType dtype,
                          DenseTensor* out) {
  std::normal_distribution<T> dist(mean, std);
  int64_t size = out->numel();
  dev_ctx.template Alloc<T>(out);
  auto* data = out->data();
  unsigned int seed = static_cast<unsigned int>(context.Attr<int>("seed"));
  // TODO(pangyoki): implement GetXPURandomEngine to set different seeds on
  // corresponding XPU device.
  auto engine = framework::GetCPURandomEngine(seed);

  std::unique_ptr<T[]> data_cpu(new T[size]);
  for (int64_t i = 0; i < size; ++i) {
    data_cpu[i] = dist(*engine);
  }
  memory::Copy(context.GetPlace(),
               data,
               platform::CPUPlace(),
               reinterpret_cast<void*>(data_cpu.get()),
               size * sizeof(T));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    gaussian_random, XPU, ALL_LAYOUT, phi::GaussianRandomKernel, float) {}

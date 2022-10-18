/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/uniform_random_kernel.h"

#include <string>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/uniform_real_distribution.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomRawKernel(const Context &dev_ctx,
                            const IntArray &shape,
                            DataType dtype,
                            const Scalar &min,
                            const Scalar &max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            DenseTensor *out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T *data = dev_ctx.template Alloc<T>(out);
  int64_t size = out->numel();

  std::unique_ptr<T[]> data_cpu(new T[size]);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  UniformRealDistribution<T>(
      data_cpu.get(), size, min.to<float>(), max.to<float>(), engine);
  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        phi::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      data_cpu[pos] = diag_val;
    }
  }

  paddle::memory::Copy(dev_ctx.GetPlace(),
                       data,
                       phi::CPUPlace(),
                       reinterpret_cast<void *>(data_cpu.get()),
                       size * sizeof(T));
}

}  // namespace phi

PD_REGISTER_KERNEL(
    uniform_random_raw, XPU, ALL_LAYOUT, phi::UniformRandomRawKernel, float) {}

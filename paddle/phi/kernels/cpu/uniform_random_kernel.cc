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

#include "paddle/phi/kernels/uniform_random_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
inline void UniformRealDistribution(T *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void UniformRealDistribution(phi::dtype::bfloat16 *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  std::uniform_real_distribution<float> dist(min, max);
  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<phi::dtype::bfloat16>(dist(*engine));
  }
}

template <typename T, typename Context>
void UniformRandomRawKernel(const Context &dev_ctx,
                            const ScalarArray &shape,
                            DataType dtype,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            DenseTensor *out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  VLOG(4) << out->dims();
  T *data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();
  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  UniformRealDistribution<T>(data, size, min, max, engine);
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
      data[pos] = diag_val;
    }
  }
}

template <typename T, typename Context>
void UniformRandomKernel(const Context &dev_ctx,
                         const ScalarArray &shape,
                         DataType dtype,
                         float min,
                         float max,
                         int seed,
                         DenseTensor *out) {
  UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(uniform_random_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomRawKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(uniform_random,
                   CPU,
                   ALL_LAYOUT,
                   phi::UniformRandomKernel,
                   float,
                   double,
                   phi::dtype::bfloat16) {}

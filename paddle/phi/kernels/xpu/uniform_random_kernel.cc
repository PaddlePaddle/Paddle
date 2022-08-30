/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const IntArray& shape,
                         DataType dtype,
                         const Scalar& min,
                         const Scalar& max,
                         int seed,
                         DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  T* data = dev_ctx.template Alloc<T>(out);
  int64_t size = out->numel();

  std::unique_ptr<T[]> data_cpu(new T[size]);
  std::uniform_real_distribution<T> dist(
      static_cast<T>(ctx.Attr<float>("min")),
      static_cast<T>(ctx.Attr<float>("max")));
  unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
  auto engine = framework::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data_cpu[i] = dist(*engine);
  }

  unsigned int diag_num = static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
  unsigned int diag_step =
      static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
  auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        platform::errors::InvalidArgument(
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

  memory::Copy(ctx.GetPlace(),
               data,
               platform::CPUPlace(),
               reinterpret_cast<void*>(data_cpu.get()),
               size * sizeof(T));
}

}  // namespace phi

REGISTER_OP_XPU_KERNEL(uniform_random,
                       paddle::operators::XPUUniformRandomKernel<float>);

// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/accuracy_check_kernel.h"

#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

static constexpr float kAtolValue = 1e-8;
static constexpr float kRtolValue = 1e-5;

bool allclose(float a, float b) {
  float left = (a > b ? a - b : b - a);
  float right = kAtolValue + (b > 0 ? kRtolValue * b : (-kRtolValue) * b);
  float diff = (left > right ? left - right : right - left);
  bool val = a == b || left <= right || diff <= 1e-10;

  return val;
}

namespace phi {

template <typename T, typename Context>
void AccuracyCheckKernel(const Context& ctx,
                         const DenseTensor& x,
                         const DenseTensor& y,
                         const std::string& fn_name,
                         int64_t res_index,
                         DenseTensor* out) {
  PADDLE_ENFORCE_EQ(x.dims(),
                    y.dims(),
                    phi::errors::PreconditionNotMet("input Dim must be equal"));

  DenseTensor x_cpu;
  DenseTensor y_cpu;

  phi::Copy(ctx, x, phi::CPUPlace(), true, &x_cpu);
  phi::Copy(ctx, y, phi::CPUPlace(), true, &y_cpu);

  bool check_result = true;

  auto x_numel = x.numel();
  for (int64_t i = 0; i < x_numel; ++i) {
    if (!allclose(x_cpu.data<T>()[i], y_cpu.data<T>()[i])) {
      check_result = false;
      VLOG(2) << "Accuracy check failed between" << x_cpu.data<T>()[i]
              << " and " << y_cpu.data<T>()[i] << " at index= " << i;
      res_index = i;
      break;
    }
  }

  PADDLE_ENFORCE_EQ(check_result,
                    true,
                    phi::errors::PreconditionNotMet(
                        "Accuracy check failed, kernel name %s, res index %d",
                        fn_name,
                        res_index));
}

}  // namespace phi

PD_REGISTER_KERNEL(accuracy_check,
                   CPU,
                   ALL_LAYOUT,
                   phi::AccuracyCheckKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t,
                   int8_t,
                   int16_t,
                   phi::float16,
                   phi::bfloat16,
                   bool) {}

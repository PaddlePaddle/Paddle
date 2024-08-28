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

#include "paddle/phi/kernels/assert_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace phi {

template <typename T, typename Context>
void AssertKernel(const Context& ctx,
                  const DenseTensor& cond,
                  const std::vector<const DenseTensor*>& data,
                  int64_t summarize) {
  bool cond_flag = cond.data<bool>()[0];
  if (cond_flag) {
    return;
  }

  paddle::funcs::TensorFormatter formatter;
  formatter.SetSummarize(summarize);

  for (size_t i = 0; i < data.size(); ++i) {
    std::string name = "data_" + std::to_string(i);
    formatter.Print(*(data[i]), name);
  }

  PADDLE_THROW(common::errors::InvalidArgument(
      "The condition of  must be true, but received false"));
}

}  // namespace phi

PD_REGISTER_KERNEL(assert, CPU, ALL_LAYOUT, phi::AssertKernel, bool) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
}

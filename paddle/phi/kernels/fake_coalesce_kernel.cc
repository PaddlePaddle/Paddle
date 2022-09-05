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

#include "paddle/phi/kernels/fake_coalesce_kernel.h"

#include <sstream>
#include <vector>
#include "glog/logging.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

#include "paddle/fluid/platform/device_memory_aligment.h"

namespace phi {

template <typename T, typename Context>
void FakeCoalesceKernel(const Context &dev_ctx,
                        const std::vector<const DenseTensor *> &input,
                        DenseTensor *output) {
  int64_t size_of_dtype = sizeof(T);
  auto dtype = input.at(0)->dtype();
  int64_t numel = 0;
  // check address
  for (size_t i = 1; i < input.size(); ++i) {
    PADDLE_ENFORCE_EQ(
        dtype,
        input.at(i)->dtype(),
        errors::InvalidArgument(
            "The DataType of input tensors of fake_coalesce should be "
            "consistent, current dtype is: %s, but the previous dtype is %s",
            dtype,
            input.at(i)->dtype()));
    const void *cur_address = input.at(i - 1)->data();
    int64_t len = input.at(i - 1)->numel();
    auto offset =
        paddle::platform::Alignment(len * size_of_dtype, dev_ctx.GetPlace());
    void *infer_next_address = reinterpret_cast<void *>(
        reinterpret_cast<uintptr_t>(cur_address) + offset);
    const void *next_address = input.at(i)->data();
    numel += offset;

    VLOG(10) << ::paddle::string::Sprintf(
        "Input[%d] address: 0X%02x, Input[%d] address: 0X%02x, Infer "
        "input[%d] address: 0X%02x, offset: %d.",
        i - 1,
        cur_address,
        i,
        next_address,
        i,
        infer_next_address,
        offset);
    PADDLE_ENFORCE_EQ(
        infer_next_address,
        next_address,
        errors::InvalidArgument(
            "The infered address of the next tensor should be equal to the "
            "real address of the next tensor. But got infered address is %p "
            "and real address is %p.",
            infer_next_address,
            next_address));
  }
  numel += paddle::platform::Alignment(
      (*input.rbegin())->numel() * size_of_dtype, dev_ctx.GetPlace());
  // reset holder, do inplace
  output->ResetHolder(input.at(0)->Holder());
  output->Resize({numel / size_of_dtype});
  output->set_type(dtype);
}

}  // namespace phi

PD_REGISTER_KERNEL(fake_coalesce,
                   CPU,
                   ALL_LAYOUT,
                   phi::FakeCoalesceKernel,
                   int,
                   float,
                   double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(fake_coalesce,
                   GPU,
                   ALL_LAYOUT,
                   phi::FakeCoalesceKernel,
                   phi::dtype::float16,
                   int,
                   float,
                   double) {}
#endif

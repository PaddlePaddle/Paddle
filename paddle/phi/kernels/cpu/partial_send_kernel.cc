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

#include <algorithm>
#include <utility>
#include <vector>

#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void PartialSendOpCPUKernel(const Context& dev_ctx UNUSED,
                            const DenseTensor& x UNUSED,
                            int ring_id UNUSED,
                            int peer UNUSED,
                            bool use_calc_stream UNUSED,
                            int num UNUSED,
                            int id UNUSED) {
  PADDLE_THROW(common::errors::Unavailable(
      "Do not support partial_send for cpu kernel now."));
}

}  // namespace phi

PD_REGISTER_KERNEL(partial_send,
                   CPU,
                   ALL_LAYOUT,
                   phi::PartialSendOpCPUKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16) {}

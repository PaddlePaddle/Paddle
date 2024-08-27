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

#include <string>
#include "glog/logging.h"
#include "paddle/phi/core/kernel_registry.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/phi/core/platform/collective_helper.h"
#endif

namespace phi {

template <typename T, typename Context>
void CommInitAllKernel(const Context& dev_ctx,
                       const std::vector<int>& devices_input,
                       int ring_id) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  std::vector<int> devices = devices_input;
  if (devices.empty()) {
    devices = phi::backends::gpu::GetSelectedDevices();
  }

  paddle::platform::NCCLCommContext::Instance().CreateAllNCCLComms(devices,
                                                                   ring_id);
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(
    comm_init_all, GPU, ALL_LAYOUT, phi::CommInitAllKernel, float) {}

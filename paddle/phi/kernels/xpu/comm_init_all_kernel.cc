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
#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/phi/core/platform/collective_helper.h"
#include "paddle/phi/core/platform/device/xpu/bkcl_helper.h"
#endif

namespace phi {

template <typename T, typename Context>
void CommInitAllKernel(const Context& dev_ctx,
                       const std::vector<int>& devices_input,
                       int ring_id) {
#if defined(PADDLE_WITH_XPU_BKCL)
  std::vector<int> devices = devices_input;

  if (devices.empty()) {
    int count = phi::backends::xpu::GetXPUDeviceCount();
    for (int i = 0; i < count; ++i) {
      devices.push_back(i);
    }
  }

  if (devices.size() > 1) {
    std::vector<phi::Place> place_list_;
    for (size_t i = 0; i < devices.size(); ++i) {
      auto p = phi::XPUPlace(devices[i]);
      place_list_.push_back(p);
    }

    // create pthread to bkcl_init_rank on all devices
    auto ptr = new paddle::platform::BKCLContextMap(place_list_);
    ptr->init();

    for (size_t i = 0; i < devices.size(); ++i) {
      paddle::platform::BKCLCommContext::Instance().AssignBKCLComm(
          ptr->contexts_.at(devices[i]).comm_,
          devices.size(),
          devices[i],
          devices[i],
          ring_id);

      VLOG(0) << "bkcl communicator of rank " << devices[i] << " in ring "
              << ring_id << " has been created on device " << devices[i];

      // TODO(WorgenZhang): need release comm_map_ when quit
      // std::call_once(once_flag_, []() {
      //   std::atexit([]() {
      //   platform::BKCLCommContext::Instance().ReleaseBKCLComms(); });
      // });
    }

    VLOG(0) << "done bkcl_init_rank on all devices";
  }
#endif
}

}  // namespace phi

PD_REGISTER_KERNEL(
    comm_init_all, XPU, ALL_LAYOUT, phi::CommInitAllKernel, float) {}

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/platform/collective_helper.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T, typename DeviceContext>
class CCommInitAllKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    std::vector<int> devices = ctx.Attr<std::vector<int>>("devices");
    if (devices.empty()) {
      devices = platform::GetSelectedDevices();
    }

    int rid = ctx.Attr<int>("ring_id");

    platform::NCCLCommContext::Instance().CreateAllNCCLComms(devices, rid);
#elif defined(PADDLE_WITH_XPU_BKCL)
    std::vector<int> devices = ctx.Attr<std::vector<int>>("devices");
    int ring_id = ctx.Attr<int>("ring_id");

    if (devices.empty()) {
      int count = platform::GetXPUDeviceCount();
      for (int i = 0; i < count; ++i) {
        devices.push_back(i);
      }
    }

    if (devices.size() > 1) {
      std::vector<platform::Place> place_list_;
      for (size_t i = 0; i < devices.size(); ++i) {
        auto p = platform::XPUPlace(devices[i]);
        place_list_.push_back(p);
      }

      // create pthread to bkcl_init_rank on all devices
      auto ptr = new platform::BKCLContextMap(place_list_);
      ptr->init();

      for (size_t i = 0; i < devices.size(); ++i) {
        platform::BKCLCommContext::Instance().AssignBKCLComm(
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
};

}  // namespace operators
}  // namespace paddle

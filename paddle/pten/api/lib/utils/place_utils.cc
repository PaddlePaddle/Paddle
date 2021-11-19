/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pten/api/lib/utils/place_utils.h"
#include "paddle/pten/api/ext/exception.h"

namespace paddle {
namespace experimental {

Place convert_to_pten_place(const platform::Place& src) {
  Place place;
  if (is_cpu_place(src)) {
    place.Reset(Device(DeviceType::HOST, 0));
  } else if (platform::is_gpu_place(src)) {
    place.Reset(
        Device(DeviceType::CUDA,
               BOOST_GET_CONST(platform::CUDAPlace, src).GetDeviceId()));
  } else if (platform::is_cuda_pinned_place(src)) {
    place.Reset(Device(DeviceType::CUDA, 0), true);
  } else if (platform::is_xpu_place(src)) {
    place.Reset(Device(DeviceType::XPU,
                       BOOST_GET_CONST(platform::XPUPlace, src).GetDeviceId()));
  } else {
    PD_THROW("Invalid platform place type.");
  }
  return place;
}

platform::Place convert_to_pd_place(const Place& src) {
  switch (src.device().type()) {
    case DeviceType::HOST: {
      return platform::CPUPlace();
    }
    case DeviceType::CUDA: {
      if (src.is_pinned()) {
        return platform::CUDAPinnedPlace();
      } else {
        return platform::CUDAPlace(src.device().id());
      }
    }
    case DeviceType::XPU: {
      return platform::XPUPlace(src.device().id());
    }
    default:
      PD_THROW("Invalid pten place type.");
  }
  return {};
}

}  // namespace experimental
}  // namespace paddle

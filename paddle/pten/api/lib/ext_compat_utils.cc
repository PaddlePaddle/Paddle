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

#include "paddle/pten/api/lib/ext_compat_utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace experimental {

platform::Place ConvertExtPlaceToPlatformPlace(const PlaceType& p) {
  if (p == PlaceType::kCPU) {
    return platform::Place(platform::CPUPlace());
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (p == PlaceType::kGPU) {
    return platform::Place(platform::CUDAPlace(platform::GetCurrentDeviceId()));
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported place type code(%d) when "
                                        "casting enum place to paddle place.",
                                        static_cast<int>(p)));
  }
  return platform::Place();
}

Place ConvertExtPlaceToInnerPlace(PlaceType p) {
  if (p == PlaceType::kCPU) {
    return Place(DeviceType::kHost);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  } else if (p == PlaceType::kGPU) {
    return Place(DeviceType::kCuda, platform::GetCurrentDeviceId());
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported place type code(%d) when "
                                        "casting enum place to paddle place.",
                                        static_cast<int>(p)));
  }
  return {};
}

PlaceType ConvertInnerPlaceToExtPlace(const Place& p) {
  if (p.device_type() == DeviceType::kHost) {
    return PlaceType::kCPU;
  } else if (p.device_type() == DeviceType::kCuda) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    return PlaceType::kGPU;
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported place type `%s` when "
                                        "casting paddle place to enum place.",
                                        p.DebugString()));
  }
  return PlaceType::kUNK;
}

Backend ConvertExtPlaceToBackend(PlaceType p) {
  switch (p) {
    case PlaceType::kCPU:
      return Backend::CPU;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case PlaceType::kGPU:
      return Backend::CUDA;
#endif
    default:
      PADDLE_THROW(
          platform::errors::Unimplemented("Unsupported place type `%s` when "
                                          "casting enum place to backend.",
                                          static_cast<int>(p)));
  }
}

}  // namespace experimental
}  // namespace paddle

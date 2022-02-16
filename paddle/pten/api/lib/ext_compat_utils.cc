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
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace experimental {

platform::Place ConvertExtPlaceToInnerPlace(PlaceType p) {
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

PlaceType ConvertInnerPlaceToExtPlace(const platform::Place& p) {
  if (platform::is_cpu_place(p)) {
    return PlaceType::kCPU;
  } else if (platform::is_gpu_place(p)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    return PlaceType::kGPU;
#endif
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported place type `%s` when "
                                        "casting paddle place to enum place.",
                                        p));
  }
  return PlaceType::kUNK;
}

Backend ConvertExtPlaceToBackend(PlaceType p) {
  switch (p) {
    case PlaceType::kCPU:
      return Backend::CPU;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case PlaceType::kGPU:
      return Backend::GPU;
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

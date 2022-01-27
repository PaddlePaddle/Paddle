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
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/core/kernel_alias_name.h"
// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace pten {

// TODO(chenweihang): Add other place trans cases later
Backend TransToPtenBackend(const paddle::platform::Place& place) {
  if (paddle::platform::is_cpu_place(place)) {
    return Backend::CPU;
  } else if (paddle::platform::is_gpu_place(place)) {
    return Backend::GPU;
  } else {
    return Backend::UNDEFINED;
  }
}

paddle::platform::Place TransToFluidPlace(const Backend& backend) {
  // TODO(chenweihang): add other trans cases later
  switch (backend) {
    case pten::Backend::CPU:
      return paddle::platform::CPUPlace();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::GPU:
      return paddle::platform::CUDAPlace(
          paddle::platform::GetCurrentDeviceId());
#endif
#ifdef PADDLE_WITH_MKLDNN
    case pten::Backend::MKLDNN:
      return paddle::platform::CPUPlace();
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    case pten::Backend::CUDNN:
      return paddle::platform::CUDAPlace(
          paddle::platform::GetCurrentDeviceId());
#endif
    default:
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Unsupported backend `%s` when casting it to paddle place type.",
          backend));
  }
}

paddle::framework::LoD TransToFluidLoD(const pten::LoD& lod) {
  paddle::framework::LoD out;
  out.reserve(lod.size());

  for (auto& elem : lod) {
    out.emplace_back(elem);
  }
  return out;
}

pten::LoD TransToPtenLoD(const paddle::framework::LoD& lod) {
  pten::LoD out;
  out.reserve(lod.size());

  for (auto& elem : lod) {
    out.emplace_back(elem);
  }
  return out;
}

const std::string& TransToPtenKernelName(const std::string& fluid_op_name) {
  if (kernel_alias_name_map.find(fluid_op_name) !=
      kernel_alias_name_map.end()) {
    return kernel_alias_name_map.at(fluid_op_name);
  }
  return fluid_op_name;
}

}  // namespace pten

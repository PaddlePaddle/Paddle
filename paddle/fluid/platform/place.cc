/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/place.h"

PADDLE_DEFINE_EXPORTED_bool(
    benchmark, false,
    "Doing memory benchmark. It will make deleting scope synchronized, "
    "and add some memory usage logs."
    "Default cuda is asynchronous device, set to True will"
    "force op run in synchronous mode.");

namespace paddle {
namespace platform {

bool is_gpu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::GPU;
}

bool is_xpu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::XPU;
}

bool is_mlu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::MLU;
}

bool is_npu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::NPU;
}

bool is_ipu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::IPU;
}

bool is_cpu_place(const Place &p) {
  return p.GetType() == phi::AllocationType::CPU;
}

bool is_cuda_pinned_place(const Place &p) {
  return p.GetType() == phi::AllocationType::GPUPINNED;
}

bool is_npu_pinned_place(const Place &p) {
  return p.GetType() == phi::AllocationType::NPUPINNED;
}

bool is_custom_place(const Place &p) {
  return p.GetType() == phi::AllocationType::CUSTOM;
}

bool places_are_same_class(const Place &p1, const Place &p2) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (is_custom_place(p1) && is_custom_place(p2)) {
    return p1.GetDeviceType() == p2.GetDeviceType();
  }
#endif
  return p1.GetType() == p2.GetType();
}

bool is_same_place(const Place &p1, const Place &p2) {
  if (places_are_same_class(p1, p2)) {
    if (is_cpu_place(p1) || is_cuda_pinned_place(p1) ||
        is_npu_pinned_place(p1)) {
      return true;
    } else if (is_xpu_place(p1)) {
      return p1 == p2;
    } else if (is_mlu_place(p1)) {
      return p1 == p2;
    } else if (is_npu_place(p1)) {
      return p1 == p2;
    } else if (is_ipu_place(p1)) {
      return p1 == p2;
    } else if (is_custom_place(p1)) {
      return p1 == p2;
    } else {
      return p1 == p2;
    }
  } else {
    return false;
  }
}

#ifdef PADDLE_WITH_CUSTOM_DEVICE
std::string PlaceHelper::GetDeviceType(const Place &place) {
  if (is_cpu_place(place)) {
    return "cpu";
  } else if (is_gpu_place(place)) {
    return "gpu";
  } else if (is_npu_place(place)) {
    return "npu";
  } else if (is_xpu_place(place)) {
    return "xpu";
  } else if (is_custom_place(place)) {
    return place.GetDeviceType();
  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "Unknown device type. Please check available devices by "
        "paddle.device.get_available_device()"));
  }
}

size_t PlaceHelper::GetDeviceId(const Place &place) {
  return place.GetDeviceId();
}

Place PlaceHelper::CreatePlace(const std::string &dev_type, size_t dev_id) {
  if (dev_type == "cpu") {
    return platform::CPUPlace();
  } else if (dev_type == "gpu") {
    return platform::CUDAPlace(dev_id);
  } else if (dev_type == "npu") {
    return platform::NPUPlace(dev_id);
  } else if (dev_type == "xpu") {
    return platform::XPUPlace(dev_id);
  } else {
    return platform::CustomPlace(dev_type, dev_id);
  }
}
#endif

}  // namespace platform
}  // namespace paddle

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
#pragma once

// #include <functional>
// #include <iostream>
// #include <vector>

#include "paddle/fluid/platform/enforce.h"
// #include "paddle/fluid/platform/variant.h"
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/enforce_npu.h"
#endif

#include "paddle/phi/common/place.h"
namespace paddle {
namespace platform {

using Place = phi::Place;
using CPUPlace = phi::CPUPlace;
using CUDAPlace = phi::GPUPlace;
using CUDAPinnedPlace = phi::GPUPinnedPlace;
using NPUPlace = phi::NPUPlace;
using NPUPinnedPlace = phi::NPUPinnedPlace;
using XPUPlace = phi::XPUPlace;
using IPUPlace = phi::IPUPlace;
using MLUPlace = phi::MLUPlace;
using CustomPlace = phi::CustomPlace;

using PlaceList = std::vector<Place>;

#ifdef PADDLE_WITH_CUSTOM_DEVICE
class PlaceHelper {
 public:
  static std::string GetDeviceType(const Place &place);
  static size_t GetDeviceId(const Place &place);
  static Place CreatePlace(const std::string &dev_type, size_t dev_id = 0);
};
#endif

bool is_gpu_place(const Place &);
bool is_xpu_place(const Place &);
bool is_npu_place(const Place &);
bool is_mlu_place(const Place &);
bool is_ipu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool is_npu_pinned_place(const Place &);
bool is_custom_place(const Place &p);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  switch (place.GetType()) {
    case phi::AllocationType::GPU: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::GPUPINNED: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::XPU: {
#ifdef PADDLE_WITH_XPU
      platform::XPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Paddle is not compiled with XPU. Cannot visit xpu device"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::NPU: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::NPUPINNED: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::IPU: {
#ifdef PADDLE_WITH_IPU
      platform::IPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with IPU. Cannot visit ipu device"));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::MLU: {
#ifdef PADDLE_WITH_MLU
      platform::MLUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with MLU. Cannot visit mlu device"));
#endif
    }
    case phi::AllocationType::CUSTOM: {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      platform::CustomPlace p(place.GetDeviceType(), place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUSTOM. Cannot visit custom device"));
#endif
    }
    default: {
      platform::CPUPlace p;
      return visitor(p);
    }
  }
}

}  // namespace platform
}  // namespace paddle

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

#include "paddle/pten/common/place.h"
namespace paddle {
namespace platform {

using Place = pten::Place;
using CPUPlace = pten::CPUPlace;
using CUDAPlace = pten::GPUPlace;
using CUDAPinnedPlace = pten::GPUPinnedPlace;
using NPUPlace = pten::NPUPlace;
using NPUPinnedPlace = pten::NPUPinnedPlace;
using XPUPlace = pten::XPUPlace;
using IPUPlace = pten::IPUPlace;
using MLUPlace = pten::MLUPlace;

using PlaceList = std::vector<Place>;

bool is_gpu_place(const Place &);
bool is_xpu_place(const Place &);
bool is_npu_place(const Place &);
bool is_mlu_place(const Place &);
bool is_ipu_place(const Place &);
bool is_cpu_place(const Place &);
bool is_cuda_pinned_place(const Place &);
bool is_npu_pinned_place(const Place &);
bool places_are_same_class(const Place &, const Place &);
bool is_same_place(const Place &, const Place &);

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  switch (place.GetType()) {
    case pten::AllocationType::GPU: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::GPUPINNED: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::CUDAPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with CUDA. Cannot visit cuda_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::XPU: {
#ifdef PADDLE_WITH_XPU
      platform::XPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(paddle::platform::errors::Unavailable(
          "Paddle is not compiled with XPU. Cannot visit xpu device"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::NPU: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::NPUPINNED: {
#ifdef PADDLE_WITH_ASCEND_CL
      platform::NPUPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with NPU. Cannot visit npu_pinned"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::IPU: {
#ifdef PADDLE_WITH_IPU
      platform::IPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with IPU. Cannot visit ipu device"));
      return typename Visitor::result_type();
#endif
    }
    case pten::AllocationType::MLU: {
#ifdef PADDLE_WITH_MLU
      platform::MLUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(platform::errors::Unavailable(
          "Paddle is not compiled with MLU. Cannot visit mlu device"));
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

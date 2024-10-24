// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/enforce.h"

namespace phi {

// need add dependency to phi_place when use phi::VisitPlace
template <typename Visitor>
typename Visitor::result_type VisitPlace(const phi::Place& place,
                                         const Visitor& visitor) {
  switch (place.GetType()) {
    case phi::AllocationType::GPU: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::GPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(common::errors::Unavailable(
          ("Paddle is not compiled with CUDA. Cannot visit cuda_pinned")));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::GPUPINNED: {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::GPUPinnedPlace p;
      return visitor(p);
#else
      PADDLE_THROW(common::errors::Unavailable(
          ("Paddle is not compiled with CUDA. Cannot visit cuda_pinned")));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::XPU: {
#ifdef PADDLE_WITH_XPU
      phi::XPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(common::errors::Unavailable(
          ("Paddle is not compiled with XPU. Cannot visit xpu device")));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::IPU: {
#ifdef PADDLE_WITH_IPU
      phi::IPUPlace p(place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(common::errors::Unavailable(
          ("Paddle is not compiled with IPU. Cannot visit ipu device")));
      return typename Visitor::result_type();
#endif
    }
    case phi::AllocationType::CUSTOM: {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
      phi::CustomPlace p(place.GetDeviceType(), place.GetDeviceId());
      return visitor(p);
#else
      PADDLE_THROW(common::errors::Unavailable(
          ("Paddle is not compiled with CUSTOM. Cannot visit custom device")));
#endif
    }
    default: {
      phi::CPUPlace p;
      return visitor(p);
    }
  }
}

}  // namespace phi

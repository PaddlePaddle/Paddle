// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <stdexcept>

#include "paddle/phi/common/place.h"

namespace compat {

inline phi::Place _PD_GetCreatePinnedPlace(const phi::Place& base_place) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#if defined(PADDLE_WITH_XPU)
  if (phi::is_xpu_place(base_place)) {
    return phi::Place(phi::XPUPinnedPlace());
  }
#endif
  return phi::Place(phi::GPUPinnedPlace());
#elif defined(PADDLE_WITH_XPU)
  (void)base_place;
  return phi::Place(phi::XPUPinnedPlace());
#else
  (void)base_place;
  throw std::runtime_error(
      "pin_memory is not supported: no GPU/XPU backend enabled");
#endif
}

}  // namespace compat

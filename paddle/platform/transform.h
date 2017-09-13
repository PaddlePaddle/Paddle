/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/platform/enforce.h"
#include "paddle/platform/hostdevice.h"
#include "paddle/platform/place.h"

#include <algorithm>
#include <type_traits>
#ifdef __NVCC__
#include <thrust/transform.h>
#include "paddle/platform/details/device_ptr_cast.h"
#endif

namespace paddle {
namespace platform {
// Transform on host or device. It provides the same API in std library.
template <typename Place, typename InputIter, typename OutputIter,
          typename UnaryOperation>
void Transform(Place place, InputIter first, InputIter last, OutputIter result,
               UnaryOperation op) {
  if (is_cpu_place(place)) {
    std::transform(first, last, result, op);
  } else {
#ifdef __NVCC__
    using namespace details;
    thrust::transform(DevPtrCast(first), DevPtrCast(last), DevPtrCast(result),
                      op);
#else
    PADDLE_THROW("Do not invoke `Transform<GPUPlace>` in .cc file");
#endif
  }
}

template <typename Place, typename InputIter1, typename InputIter2,
          typename OutputIter, typename BinaryOperation>
void Transform(Place place, InputIter1 first1, InputIter1 last1,
               InputIter2 first2, OutputIter result, BinaryOperation op) {
  if (is_cpu_place(place)) {
    std::transform(first1, last1, first2, result, op);
  } else {
#ifdef __NVCC__
    using namespace details;
    thrust::transform(DevPtrCast(first1), DevPtrCast(last1), DevPtrCast(first2),
                      DevPtrCast(result), op);
#else
    PADDLE_THROW("Do not invoke `Transform<GPUPlace>` in .cc file");
#endif
  }
};

}  // namespace platform
}  // namespace paddle

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

#ifndef __NVCC__
#error device_ptr_cast must be include by .cu file
#endif

#include <thrust/device_ptr.h>

namespace paddle {
namespace platform {
namespace details {
template <typename T, bool is_ptr>
struct DevicePtrCast;

template <typename T>
struct DevicePtrCast<T, true> {
  using ELEM = typename std::remove_pointer<T>::type;
  using RTYPE = thrust::device_ptr<ELEM>;

  inline thrust::device_ptr<ELEM> operator()(ELEM* ele) const {
    return thrust::device_pointer_cast(ele);
  }
};

template <typename T>
struct DevicePtrCast<T, false> {
  using RTYPE = T;
  inline RTYPE operator()(RTYPE it) const { return it; }
};

// Cast T to thrust::device_ptr if T is a pointer.
// Otherwise, e.g., T is a iterator, return T itself.
template <typename T>
auto DevPtrCast(T t) ->
    typename DevicePtrCast<T, std::is_pointer<T>::value>::RTYPE {
  DevicePtrCast<T, std::is_pointer<T>::value> cast;
  return cast(t);
}

}  // namespace details
}  // namespace platform
}  // namespace paddle

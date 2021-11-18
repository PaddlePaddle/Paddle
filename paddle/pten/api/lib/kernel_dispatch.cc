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

#include "paddle/pten/api/lib/kernel_dispatch.h"

#include "paddle/pten/core/convert_utils.h"

namespace paddle {
namespace experimental {
namespace detail {

BackendSet GetTensorBackendSet(const Tensor& t) {
  BackendSet backend_set(pten::TransToPtenBackend(t.inner_place()));
  switch (t.layout()) {
    case DataLayout::MKLDNN:
      backend_set = backend_set | BackendSet(Backend::MKLDNN);
      break;
    default:
      // do nothing
      break;
  }
  return backend_set;
}

std::size_t CountLeadingZeros(uint64_t val) {
  if (val == 0) {
    return 64;
  }
  std::size_t zero_bits = 0;
  for (std::size_t shift = 64 >> 1; shift; shift >>= 1) {
    uint64_t tmp = val >> shift;
    if (tmp) {
      val = tmp;
    } else {
      zero_bits |= shift;
    }
  }
  return zero_bits;
}

}  // namespace detail

paddle::platform::DeviceContext* GetDeviceContextByBackend(
    pten::Backend backend) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  return pool.Get(pten::TransToFluidPlace(backend));
}

}  // namespace experimental
}  // namespace paddle

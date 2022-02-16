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

#include "paddle/pten/core/compat/convert_utils.h"

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

pten::DeviceContext* GetDeviceContextByBackend(pten::Backend backend) {
  auto& pool = paddle::platform::DeviceContextPool::Instance();
  return pool.Get(pten::TransToPtenPlace(backend));
}

DataType ParseDataType(DataType dtype) { return dtype; }
DataType ParseDataType(const Tensor& tensor) { return tensor.type(); }
DataType ParseDataType(const std::vector<Tensor>& tensors) {
  if (tensors.empty()) {
    return DataType::UNDEFINED;
  }
  DataType dtype = tensors[0].type();
  auto n = tensors.size();
  for (size_t i = 1; i < n; ++i) {
    if (tensors[i].type() != dtype) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The data_type of input tensor in list isn't consistent, "
          "the first tensor is %s, but %dth tensor is %s.",
          dtype,
          i,
          tensors[i].type()));
    }
  }
  return dtype;
}

DataType ParseDataTypeWithInputOrder(DataType dtype, const Tensor& tensor) {
  return dtype != DataType::UNDEFINED ? dtype : ParseDataType(tensor);
}

Backend ParseBackend(Backend backend) { return backend; }
Backend ParseBackend(const Tensor& tensor) {
  return pten::TransToPtenBackend(tensor.inner_place());
}

Backend ParseBackendWithInputOrder(Backend backend, const Tensor& tensor) {
  return backend != Backend::UNDEFINED ? backend : ParseBackend(tensor);
}

DataLayout ParseLayout(DataLayout layout) { return layout; }
DataLayout ParseLayout(const Tensor& tensor) { return tensor.layout(); }

DataLayout ParseLayoutWithInputOrder(DataLayout layout, const Tensor& tensor) {
  return layout != DataLayout::UNDEFINED ? layout : ParseLayout(tensor);
}

}  // namespace experimental
}  // namespace paddle

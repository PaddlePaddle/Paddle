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

#include "paddle/phi/api/lib/kernel_dispatch.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_tensor.h"
#include "paddle/phi/core/string_tensor_utils.h"
#include "paddle/phi/core/tensor_utils.h"
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif

namespace paddle {
namespace experimental {
namespace detail {

// We need judge whether the allocation is nullptr,
// whether the allocation is initialized, wo we need GetHolder method
bool HasAllocation(const phi::TensorBase& t) {
  if (phi::DenseTensor::classof(&t)) {
    return phi::DenseTensorUtils::GetHolder(
               static_cast<const phi::DenseTensor&>(t)) != nullptr;
  } else if (phi::SelectedRows::classof(&t)) {
    return phi::DenseTensorUtils::GetHolder(
               static_cast<const phi::SelectedRows&>(t).value()) != nullptr;
  } else if (phi::SparseCsrTensor::classof(&t)) {
    return phi::DenseTensorUtils::GetHolder(
               static_cast<const phi::SparseCsrTensor&>(t)
                   .non_zero_elements()) != nullptr;
  } else if (phi::SparseCooTensor::classof(&t)) {
    return phi::DenseTensorUtils::GetHolder(
               static_cast<const phi::SparseCooTensor&>(t)
                   .non_zero_elements()) != nullptr;
  } else if (phi::StringTensor::classof(&t)) {
    return phi::StringTensorUtils::GetHolder(
               static_cast<const phi::StringTensor&>(t)) != nullptr;
  } else if (phi::distributed::DistTensor::classof(&t)) {
    return static_cast<const phi::distributed::DistTensor&>(t).defined();
  } else {
    return false;
  }
}

BackendSet GetTensorBackendSet(const phi::TensorBase& t) {
  if (HasAllocation(t) && t.place().GetType() != AllocationType::UNDEFINED) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    // See Note [ Why `SetDevice` when parsing custom place? ]
    if (t.place().GetType() == AllocationType::CUSTOM) {
      phi::DeviceManager::SetDevice(t.place());
    }
#endif
    phi::Backend backend_key = phi::TransToPhiBackend(t.place());
    BackendSet backend_set(backend_key);
    if (backend_key == Backend::GPU && phi::DenseTensor::classof(&t) &&
        static_cast<const phi::DenseTensor&>(t).meta().use_gpudnn) {
      backend_set = backend_set | BackendSet(Backend::GPUDNN);
    }
    return backend_set;
  }
  return BackendSet(Backend::UNDEFINED);
}

std::size_t CountLeadingZeros(uint32_t val) {
#if defined(__clang__) || defined(__GNUC__)
  return __builtin_clz(val);
#elif defined(_MSC_VER)
  // windows don't have built-in clz/ctz function
  DWORD Index;
  _BitScanReverse(&Index, val);
  return (uint32_t)Index ^ 31;
#else
  if (val == 0) {
    return 32;
  }
  std::size_t zero_bits = 0;
  for (std::size_t shift = 32 >> 1; shift; shift >>= 1) {
    uint32_t tmp = val >> shift;
    if (tmp) {
      val = tmp;
    } else {
      zero_bits |= shift;
    }
  }
  return zero_bits;
#endif
}

}  // namespace detail

phi::DeviceContext* GetDeviceContextByBackend(phi::Backend backend) {
  auto& pool = paddle::experimental::DeviceContextPool::Instance();
  return pool.GetMutable(phi::TransToPhiPlace(backend));
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
      PADDLE_THROW(phi::errors::InvalidArgument(
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

Backend ParseBackend(const Place& place) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  /**
   * [ Why `SetDevice` when parsing custom place? ]
   * Users are able to call C++ APIs under customOP + customDevice scenario. To
   * make sure `GetDevice` function outputs the accurate place when executing
   * `GetDeviceContextByBackend` function in C++ API, we need to call
   * `SetDevice` first. However, in dygraph mode, `SetDevice` is called at
   * CPython level and calling C++ API directly in customOP cannot reach
   * CPython. Hence, we need to manually set the device here.
   */
  if (place.GetType() == AllocationType::CUSTOM) {
    phi::DeviceManager::SetDevice(place);
  }
#endif
  return phi::TransToPhiBackend(place);
}
Backend ParseBackend(const Tensor& tensor) {
  Backend backend_key = phi::TransToPhiBackend(tensor.place());
  if (backend_key == Backend::GPU &&
      phi::DenseTensor::classof(tensor.impl().get()) &&
      static_cast<phi::DenseTensor*>(tensor.impl().get())->meta().use_gpudnn) {
    return Backend::GPUDNN;
  }
  return backend_key;
}

Backend ParseBackendWithInputOrder(const Place& place, const Tensor& tensor) {
  return place.GetType() != phi::AllocationType::UNDEFINED
             ? ParseBackend(place)
             : ParseBackend(tensor);
}

DataLayout ParseLayout(DataLayout layout) { return layout; }
DataLayout ParseLayout(const Tensor& tensor) { return tensor.layout(); }

DataLayout ParseLayoutWithInputOrder(DataLayout layout, const Tensor& tensor) {
  return layout != DataLayout::UNDEFINED ? layout : ParseLayout(tensor);
}

}  // namespace experimental
}  // namespace paddle

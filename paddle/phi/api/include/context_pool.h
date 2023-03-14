/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <mutex>

#include "paddle/phi/api/include/dll_decl.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/macros.h"
#include "paddle/utils/flat_hash_map.h"

namespace phi {
class DeviceContext;
class CPUContext;
class GPUContext;
class Allocator;
class CUDAStream;
}  // namespace phi

namespace paddle {
namespace experimental {

template <AllocationType T>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<AllocationType::CPU> {
  using TYPE = phi::CPUContext;
};

template <>
struct DefaultDeviceContextType<AllocationType::GPU> {
  using TYPE = phi::GPUContext;
};

/**
 * The DeviceContextPool here is just a mirror of the DeviceContextPool in
 * fluid, and does not manage the life cycle of the DeviceContext.
 * It is mainly used for external custom operator calls and high-performance
 * C++ APIs.
 *
 * Since DeviceContextPool in fluid is a global singleton, it always exists
 * in program running, so DeviceContextPool here can always access the correct
 * DeviceContext pointer.
 *
 * In order not to depend on the fluid's DeviceContextPool,
 * the DeviceContextPool here needs to be initialized in the fluid, and cannot
 * be initialized by itself.
 *
 * Note: DeviceContextPool is an experimental API and may be removed in the
 * future. From 2.3, we recommend directly using the C++ API to combine new
 * operators.
 */
class PADDLE_API DeviceContextPool {
 public:
  static DeviceContextPool& Instance();

  const phi::DeviceContext* Get(const Place& place);

  phi::DeviceContext* GetMutable(const Place& place);

  template <AllocationType T>
  const typename DefaultDeviceContextType<T>::TYPE* Get(const Place& place) {
    return reinterpret_cast<const typename DefaultDeviceContextType<T>::TYPE*>(
        Get(place));
  }

 private:
  DeviceContextPool() = default;

  paddle::flat_hash_map<Place, const phi::DeviceContext*, Place::Hash>
      context_map_;
  std::mutex mutex_;

  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace experimental
}  // namespace paddle

namespace paddle {

/**
 * Get the Allocator for the passed place.
 */
PADDLE_API phi::Allocator* GetAllocator(const phi::Place& place);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
/**
 * Get the current CUDA stream for the passed CUDA device.
 */
PADDLE_API phi::CUDAStream* GetCurrentCUDAStream(const phi::Place& place);
#endif

}  // namespace paddle

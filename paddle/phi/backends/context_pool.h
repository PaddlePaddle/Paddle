/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include <future>  // NOLINT
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>

#include "paddle/common/macros.h"
#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/utils/test_macros.h"

namespace phi {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void SetAllowTF32Cublas(bool active);
/*Get the global variable allow_tf32_cublas value*/
bool AllowTF32Cublas();
extern bool allow_tf32_cudnn;
/*Set the value of the global variable allow_tf32_cudnn*/
void SetAllowTF32Cudnn(bool active);
/*Get the global variable allow_tf32_cudnn value*/
bool AllowTF32Cudnn();
#endif  // PADDLE_WITH_CUDA

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<phi::CPUPlace> {
  using TYPE = phi::CPUContext;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <>
struct DefaultDeviceContextType<phi::GPUPlace> {
  using TYPE = phi::GPUContext;
};
#endif

#ifdef PADDLE_WITH_XPU
template <>
struct DefaultDeviceContextType<phi::XPUPlace> {
  using TYPE = phi::XPUContext;
};
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <>
struct DefaultDeviceContextType<phi::CustomPlace> {
  using TYPE = phi::CustomContext;
};
#else
template <>
struct DefaultDeviceContextType<phi::CustomPlace> {
  using TYPE = DeviceContext;
};
#endif

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  TEST_API static DeviceContextPool& Instance();

  /*! \brief  Create should only called by Init function */
  TEST_API static DeviceContextPool& Init(
      const std::vector<phi::Place>& places);

  TEST_API static bool IsInitialized();

  TEST_API static void SetPool(DeviceContextPool* dev_pool);

  /*! \brief  Return handle of single device context. */
  TEST_API phi::DeviceContext* Get(const phi::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

  TEST_API size_t Size() const;

  TEST_API const
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>&
      device_contexts() const;

  TEST_API static void SetDeviceContexts(
      const std::map<Place,
                     std::shared_future<std::unique_ptr<DeviceContext>>>*);

 private:
  explicit DeviceContextPool(const std::vector<phi::Place>& places);

  std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>
      device_contexts_;
  static thread_local const std::
      map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
          external_device_contexts_;  // not owned

  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};

}  // namespace phi

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

#include <future>  // NOLINT
#include <unordered_map>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/macros.h"
#include "paddle/phi/core/stream.h"

namespace phi {

struct MemoryInterface {
  /**
   * @brief Allocate a unique allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   */
  Allocator::AllocationPtr (*alloc)(const phi::Place& place, size_t size);

  /**
   * @brief Allocate a unique allocation.
   *
   * @param[phi::Place] place     The target gpu place that will be allocated
   * @param[size_t]     size      memory size
   * @param[phi::Stream]stream    the stream that is used for allocator
   */

  Allocator::AllocationPtr (*alloc_with_stream)(const phi::GPUPlace& place,
                                                size_t size,
                                                const phi::Stream& stream);

  /**
   * @brief Allocate a shared allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   */
  std::shared_ptr<Allocation> (*alloc_shared)(const phi::Place& place,
                                              size_t size);

  /**
   * @brief Allocate a shared allocation.
   *
   * @param[phi::Place] place     The target place that will be allocated
   * @param[size_t]     size      memory size
   * @param[phi::Stream]stream    the stream that is used for allocator
   */
  std::shared_ptr<Allocation> (*alloc_shared_with_stream)(
      const phi::Place& place, size_t size, const phi::Stream& stream);

  /**
   * @brief whether the allocation is in the stream
   *
   * @param[Allocation] allocation  the allocation to check
   * @param[phi::Stream]stream      the device's stream
   */
  bool (*in_same_stream)(const std::shared_ptr<Allocation>& allocation,
                         const phi::Stream& stream);

  /**
   * @brief free allocation
   *
   * @param[Allocation] allocation  the allocation to be freed
   */
  void (*allocation_deleter)(Allocation* allocation);

  /**
   * @brief   Copy memory from one place to another place.
   *
   * @param[Place]  DstPlace Destination allocation place (CPU or GPU or XPU or
   * CustomDevice).
   * @param[void*]  dst      Destination memory address.
   * @param[Place]  SrcPlace Source allocation place (CPU or GPU or XPU or
   * CustomDevice).
   * @param[void*]  src      Source memory address.
   * @param[size_t]  num      memory size in bytes to copy.
   * @param[void*]  stream   stream for asynchronously memory copy.
   *
   * @note    For GPU/XPU/CustomDevice memory copy, stream need to be specified
   *          for asynchronously memory copy, and type is restored in the
   *          implementation.
   *
   */
  void (*copy)(
      Place dst_place, void* dst, Place src_place, const void* src, size_t num);
  void (*copy_with_stream)(Place dst_place,
                           void* dst,
                           Place src_place,
                           const void* src,
                           size_t num,
                           void* stream);

  /**
   * @brief get the device STAT value
   *
   * @param[std::string] stat_type  memory's stat type, can be 'Allocated' or
   * 'Reserved'
   * @param[int]stream   device id
   */
  int64_t (*device_memory_stat_current_value)(const std::string& stat_type,
                                              int dev_id);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  /**
   * @brief get the memory usage of current GPU device.
   *
   * @param[size_t] available  device available memory to alloc
   * @param[size_t] total      device total memory
   */
  void (*gpu_memory_usage)(size_t* available, size_t* total);
#endif

  /**
   * @brief init devices info and device context
   */
  void (*init_devices)();

  /**
   * @brief create device_context by places and put them into
   * place_to_device_context
   *
   * @param place_to_device_context the destination that device_context will be
   * stored
   * @param places the places that are related to device_context
   * @param disable_setting_default_stream_for_allocator whether set default
   * stream for allocator
   * @param stream_priority set stream priority
   */
  void (*emplace_device_contexts)(
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
          place_to_device_context,
      const std::vector<phi::Place>& places,
      bool disable_setting_default_stream_for_allocator,
      int stream_priority);
};

class MemoryUtils {
 public:
  static MemoryUtils& Instance() {
    static MemoryUtils g_memory_utils;
    return g_memory_utils;
  }

  void Init(std::unique_ptr<MemoryInterface> memory_method) {
    memory_method_ = std::move(memory_method);
  }

  Allocator::AllocationPtr Alloc(const phi::GPUPlace& place,
                                 size_t size,
                                 const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->alloc_with_stream,
                      nullptr,
                      phi::errors::Unavailable(
                          "alloc_with_stream method in memory_method_ is not "
                          "initiazed yet. You need init it first."));
    return memory_method_->alloc_with_stream(place, size, stream);
  }

  Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->alloc,
        nullptr,
        phi::errors::Unavailable("alloc method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    return memory_method_->alloc(place, size);
  }

  std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                          size_t size,
                                          const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->alloc_shared_with_stream,
                      nullptr,
                      phi::errors::Unavailable(
                          "alloc_shared_with_stream method in memory_method_ "
                          "is not initiazed yet. You need init it first."));
    return memory_method_->alloc_shared_with_stream(place, size, stream);
  }

  std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                          size_t size) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->alloc_shared,
        nullptr,
        phi::errors::Unavailable("alloc_shared method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    return memory_method_->alloc_shared(place, size);
  }

  bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                    const phi::Stream& stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->in_same_stream,
        nullptr,
        phi::errors::Unavailable("in_same_stream method in memory_method_ is "
                                 "not initiazed yet. You need init it first."));
    return memory_method_->in_same_stream(allocation, stream);
  }

  void AllocationDeleter(Allocation* allocation) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->allocation_deleter,
                      nullptr,
                      phi::errors::Unavailable(
                          "allocation_deleter method in memory_method_ is not "
                          "initiazed yet. You need init it first."));
    return memory_method_->allocation_deleter(allocation);
  }

  void Copy(const Place& dst_place,
            void* dst,
            const Place& src_place,
            const void* src,
            size_t num,
            void* stream) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(memory_method_->copy_with_stream,
                      nullptr,
                      phi::errors::Unavailable(
                          "copy_with_stream method in memory_method_ is not "
                          "initiazed yet. You need init it first."));
    memory_method_->copy_with_stream(
        dst_place, dst, src_place, src, num, stream);
  }

  void Copy(const Place& dst_place,
            void* dst,
            const Place& src_place,
            const void* src,
            size_t num) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->copy,
        nullptr,
        phi::errors::Unavailable("copy method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    memory_method_->copy(dst_place, dst, src_place, src, num);
  }

  int64_t DeviceMemoryStatCurrentValue(const std::string& stat_type,
                                       int dev_id) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->device_memory_stat_current_value,
        nullptr,
        phi::errors::Unavailable(
            "device_memory_stat_current_value method in memory_method_ is not "
            "initiazed yet. You need init it first."));
    return memory_method_->device_memory_stat_current_value(stat_type, dev_id);
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void GpuMemoryUsage(size_t* available, size_t* total) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NOT_NULL(
        memory_method_->gpu_memory_usage,
        phi::errors::Unavailable(
            "gpu_memory_usage method in memory_method_ is not initiazed "
            "yet. You need init it first."));
    return memory_method_->gpu_memory_usage(available, total);
  }
#endif

  void InitDevices() {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->init_devices,
        nullptr,
        phi::errors::Unavailable("init_devices method in memory_method_ is not "
                                 "initiazed yet. You need init it first."));
    memory_method_->init_devices();
  }

  void EmplaceDeviceContexts(
      std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
          place_to_device_context,
      const std::vector<phi::Place>& places,
      bool disable_setting_default_stream_for_allocator,
      int stream_priority) {
    CheckMemoryMethod();
    PADDLE_ENFORCE_NE(
        memory_method_->emplace_device_contexts,
        nullptr,
        phi::errors::Unavailable(
            "emplace_device_contexts method in memory_method_ is not "
            "initiazed yet. You need init it first."));
    memory_method_->emplace_device_contexts(
        place_to_device_context,
        places,
        disable_setting_default_stream_for_allocator,
        stream_priority);
  }

  void CheckMemoryMethod() {
    PADDLE_ENFORCE_NE(
        memory_method_.get(),
        nullptr,
        phi::errors::Unavailable(
            "memory_method_ in MemoryUtils is not "
            "initiazed yet. You need init it first. If you compiled with "
            "Fluid. You can call InitMemoryMethod() for initialization."));
  }

 private:
  MemoryUtils() = default;

  std::unique_ptr<MemoryInterface> memory_method_ = nullptr;

  DISABLE_COPY_AND_ASSIGN(MemoryUtils);
};

/*
  NOTE(YuanRisheng) Why should we add the following code?
  We need this because MemoryUtils::instance() is a singleton object and we
  don't recommend using singleton object in kernels. So, we wrap it using a
  function and if we delete this singleton object in future, it will be easy to
  change code.
*/

namespace memory_utils {

Allocator::AllocationPtr Alloc(const phi::GPUPlace& place,
                               size_t size,
                               const phi::Stream& stream);

Allocator::AllocationPtr Alloc(const phi::Place& place, size_t size);

std::shared_ptr<Allocation> AllocShared(const phi::Place& place,
                                        size_t size,
                                        const phi::Stream& stream);

std::shared_ptr<Allocation> AllocShared(const phi::Place& place, size_t size);

bool InSameStream(const std::shared_ptr<Allocation>& allocation,
                  const phi::Stream& stream);

void AllocationDeleter(Allocation* allocation);

void Copy(const Place& dst_place,
          void* dst,
          const Place& src_place,
          const void* src,
          size_t num,
          void* stream);
void Copy(const Place& dst_place,
          void* dst,
          const Place& src_place,
          const void* src,
          size_t num);

int64_t DeviceMemoryStatCurrentValue(const std::string& stat_type, int dev_id);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
void GpuMemoryUsage(size_t* available, size_t* total);
#endif

void InitDevices();

void EmplaceDeviceContexts(
    std::map<Place, std::shared_future<std::unique_ptr<DeviceContext>>>*
        place_to_device_context,
    const std::vector<phi::Place>& places,
    bool disable_setting_default_stream_for_allocator,
    int stream_priority);

class Buffer {
 public:
  explicit Buffer(const phi::Place& place) : place_(place) {}

  template <typename T>
  T* Alloc(size_t size) {
    using AllocT = typename std::
        conditional<std::is_same<T, void>::value, uint8_t, T>::type;
    if (UNLIKELY(size == 0)) return nullptr;
    size *= sizeof(AllocT);
    if (allocation_ == nullptr || allocation_->size() < size) {
      allocation_ = memory_utils::Alloc(place_, size);
    }
    return reinterpret_cast<T*>(allocation_->ptr());
  }

  template <typename T>
  const T* Get() const {
    return reinterpret_cast<const T*>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  template <typename T>
  T* GetMutable() {
    return reinterpret_cast<T*>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  size_t Size() const { return allocation_ ? allocation_->size() : 0; }

  phi::Place GetPlace() const { return place_; }

 private:
  Allocator::AllocationPtr allocation_;
  phi::Place place_;
};

}  // namespace memory_utils

}  // namespace phi

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

#include <memory>

#include "paddle/common/macros.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/distributed/comm_context.h"
#include "paddle/phi/core/generator.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/core/utils/type_registry.h"

namespace phi {
class TensorBase;

/**
 * DeviceContext provides device-related interfaces.
 *
 * All kernels must access the interfaces provided by the backend through
 * DeviceContext.
 */
class PADDLE_API DeviceContext {
 public:
  /**
   * @brief Default construct.
   */
  DeviceContext();

  /**
   * @brief Copy construct.
   */
  DeviceContext(const DeviceContext&);

  /**
   * @brief Move construct.
   */
  DeviceContext(DeviceContext&&) noexcept;

  /**
   * @brief Move assign operator.
   */
  DeviceContext& operator=(DeviceContext&&) noexcept;

  /**
   * @brief Default destruct.
   */
  virtual ~DeviceContext();

  /**
   * @brief Set the device-related Allocator object.
   *
   * @param allocator
   */
  void SetAllocator(const Allocator*);

  /**
   * @brief Set the host Allocator object.
   *
   * @param allocator
   */
  void SetHostAllocator(const Allocator*);

  /**
   * @brief Set the zero-size Allocator object.
   *
   * @param allocator
   */
  void SetZeroAllocator(const Allocator*);

  /**
   * @brief Set the zero-size host Allocator object.
   *
   * @param allocator
   */
  void SetHostZeroAllocator(const Allocator*);

  /**
   * @brief Set the zero-size Allocator object.
   *
   * @param allocator
   */
  void SetPinnedAllocator(const Allocator*);

  /**
   * @brief Get the const Allocator object.
   *
   * @return Allocator
   */
  const Allocator& GetAllocator() const;

  /**
   * @brief Get the const device-related Allocator object.
   *
   * @return Allocator
   */
  const Allocator& GetHostAllocator() const;

  const Allocator& GetZeroAllocator() const;

  const Allocator& GetHostZeroAllocator() const;

  const Allocator& GetPinnedAllocator() const;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  /**
   * @brief Set the CUDA graph Allocator object.
   *
   * @param allocator
   */
  void SetCUDAGraphAllocator(const Allocator*);

  /**
   * @brief Get the const CUDA graph Allocator object.
   *
   * @return Allocator
   */
  const Allocator& GetCUDAGraphAllocator() const;

  /**
   * @brief Test whether the CUDA graph allocator is valid
   *
   * This method should be called before calling GetCUDAGraphAllocator().
   * Other unit can calls GetCUDAGraphAllocator() method,
   * only when this method returns True!
   *
   * @return true if cuda_graph_allocator_ is valid, false otherwise
   */
  bool IsCUDAGraphAllocatorValid() const;
#endif

  /**
   * @brief Allocate device memory for tensor.
   */
  virtual void* Alloc(TensorBase*,
                      DataType dtype,
                      size_t requested_size = 0,
                      bool pinned = false,
                      bool fake_alloc = false) const;

  template <typename T>
  TEST_API T* Alloc(TensorBase* tensor,
                    size_t requested_size = 0,
                    bool pinned = false) const;

  /**
   * @brief Allocate host memory for tensor.
   */
  void* HostAlloc(TensorBase* tensor,
                  DataType dtype,
                  size_t requested_size = 0,
                  bool fake_alloc = false) const;

  template <typename T>
  T* HostAlloc(TensorBase* tensor, size_t requested_size = 0) const;

  virtual const Place& GetPlace() const = 0;

  // TODO(wilber): The fluid framework uses wait() in many places, how to delete
  // this API interface.
  virtual void Wait() const {}

  /**
   * @brief Set the generator for special op.
   *
   * @param Generator
   */
  void SetGenerator(Generator*);
  /**
   * @brief Get the generator object.
   *
   * @return Generator
   */
  Generator* GetGenerator() const;

  /**
   * @brief Set the host generator for special op.
   *
   * @param Generator
   */
  void SetHostGenerator(Generator*);
  /**
   * @brief Get the host generator object.
   *
   * @return Generator
   */
  Generator* GetHostGenerator() const;

  /**
   * @brief Return the type information of the derived class to support
   *        safely downcast in non-rtti environment.
   *
   * @return The type information of the derived class.
   */
  TypeInfo<DeviceContext> type_info() const { return type_info_; }

  /**
   * @brief Set the comm context point.
   *
   * @param CommContext
   */
  void SetCommContext(distributed::CommContext* comm_context);

  /**
   * @brief Get the comm context point.
   *
   * @return comm context point
   */
  distributed::CommContext* GetCommContext() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  template <typename T, typename U>
  friend class TypeInfoTraits;
  TypeInfo<DeviceContext> type_info_{TypeInfo<DeviceContext>::kUnknownType};
};

}  // namespace phi

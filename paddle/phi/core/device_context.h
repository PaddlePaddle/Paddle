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

#include "paddle/phi/api/include/dll_decl.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/generator.h"
<<<<<<< HEAD
#include "paddle/phi/core/utils/type_registry.h"
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace phi {
class TensorBase;

/**
 * DeviceContext provides device-related interfaces.
 *
 * All kernels must access the interfaces provided by the backend through
 * DeviceContext.
 */
class PADDLE_API DeviceContext {
  using DataType = paddle::experimental::DataType;

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
  DeviceContext(DeviceContext&&);

  /**
   * @brief Move assign operator.
   */
  DeviceContext& operator=(DeviceContext&&);

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
<<<<<<< HEAD
   * @brief Set the zero-size host Allocator object.
   *
   * @param allocator
   */
  void SetHostZeroAllocator(const Allocator*);

  /**
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
  const Allocator& GetHostZeroAllocator() const;

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  const Allocator& GetPinnedAllocator() const;

#ifdef PADDLE_WITH_CUDA
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
  void* Alloc(TensorBase*,
              DataType dtype,
              size_t requested_size = 0,
<<<<<<< HEAD
              bool pinned = false,
              bool fake_alloc = false) const;
=======
              bool pinned = false) const;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

  template <typename T>
  T* Alloc(TensorBase* tensor,
           size_t requested_size = 0,
           bool pinned = false) const;

  /**
   * @brief Allocate host memory for tensor.
   */
  void* HostAlloc(TensorBase* tensor,
                  DataType dtype,
<<<<<<< HEAD
                  size_t requested_size = 0,
                  bool fake_alloc = false) const;
=======
                  size_t requested_size = 0) const;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

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

<<<<<<< HEAD
  /**
   * @brief Return the type information of the derived class to support
   *        safely downcast in non-rtti environment.
   *
   * @return The type information of the derived class.
   */
  TypeInfo<DeviceContext> type_info() const { return type_info_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;

  template <typename T, typename U>
  friend class TypeInfoTraits;
  TypeInfo<DeviceContext> type_info_{TypeInfo<DeviceContext>::kUnknownType};
=======
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
};

}  // namespace phi

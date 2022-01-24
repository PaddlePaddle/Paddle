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

// TODO(wilber): Do we need to use place in pten kernel?
#include "paddle/pten/common/place.h"

#include "paddle/pten/core/candidate/allocator.h"

namespace pten {
class TensorBase;

/**
 * DeviceContext provides device-related interfaces.
 *
 * All kernels must access the interfaces provided by the backend through
 * DeviceContext.
 */
class DeviceContext {
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
   * @brief Default destruct.
   */
  virtual ~DeviceContext();

  /**
   * @brief Set the deveice-releated Allocator object.
   *
   * @param allocator
   */
  void SetAllocator(Allocator*);

  /**
   * @brief Get the const Allocator object.
   *
   * @return Allocator
   */
  const Allocator& GetAllocator() const;

  /**
   * @brief Allocate memory for tensor.
   */
  void Alloc(pten::TensorBase*);

  // TODO(wilber): Just for the convenience of migrating the code, it will be
  // modified or removed later.
  virtual Place GetPlace() const = 0;
  // TODO(wilber): The fluid framework uses wait() in many places, how to delete
  // this API interface.
  virtual void Wait() const {}

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pten

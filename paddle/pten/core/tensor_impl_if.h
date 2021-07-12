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

#pragma once

#include "paddle/pten/core/backend.h"
#include "paddle/pten/core/ddim.h"
#include "paddle/pten/core/dtype.h"
#include "paddle/pten/core/layout.h"

namespace paddle {
namespace framework {
class DDim;
}
namespace platform {
class Place;
}
}

namespace pt {

// TODO(chenweihang): DDim still link to framework, design abstract interface
// of DDim?
using DDim = paddle::framework::DDim;

// TODO(chenweihang): Place still link to framework, design abstract interface
// of place?
using Place = paddle::platform::Place;

/**
 * The abstract class of Tensor implemention, it needs to define its basic
 * behavior through inherited classes.
 *
 */
class TensorImplInterface {
 public:
  // Not allowed to initialize a tensor without descriptive metadata
  TensorImplInterface() = default;

  TensorImplInterface(const TensorImplInterface&) = delete;
  TensorImplInterface& operator=(const TensorImplInterface&) = delete;
  TensorImplInterface(TensorImplInterface&&) = delete;
  TensorImplInterface& operator=(TensorImplInterface&&) = delete;

  virtual ~TensorImplInterface() {}

  /**
   * Most of Tensor's methods need to have corresponding implementations
   * in TensorImplInterface
   */
  virtual int64_t numel() const = 0;

  virtual DDim dims() const = 0;

  virtual void resize(const DDim& dims) = 0;

  virtual DataType type() const = 0;

  virtual Layout layout() const = 0;

  virtual Place place() const = 0;

  virtual Backend backend() const = 0;

  virtual const void* data() const = 0;

  virtual void* mutable_data() = 0;

  virtual bool initialized() const = 0;

  /**
   * template methods can not be virtual
   */
  template <typename T>
  const T* data() const {
    static_assert(std::is_pod<T>::value,
                  "T must be POD when call Tensor.data<T>().");
    return reinterpret_cast<const T*>(data());
  }

  template <typename T>
  T* mutable_data() {
    static_assert(std::is_pod<T>::value,
                  "T must be POD when call Tensor.mutable_data<T>().");
    return reinterpret_cast<T*>(mutable_data());
  }
};

}  // namespace pt

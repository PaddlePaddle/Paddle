// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <cstring>
#include <iostream>

#include "paddle/infrt/common/shared.h"

namespace infrt {
namespace common {

template <typename T>
class Shared;
/**
 * Object is the basic element in the INFRT, with `Shared` wrapper, the object
 * can be shared accross the system.
 */
struct Object {
  //! Get the type representation of this object.
  virtual const char* type_info() const = 0;
  virtual ~Object() {}

  //! Cast to a derived type.
  template <typename T>
  T* as() {
    return static_cast<T*>(this);
  }

  //! Cast to a derived type.
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(this);
  }

  //! Type safe cast.
  template <typename T>
  T* safe_as() {
    if (std::strcmp(type_info(), T::__type_info__) == 0) {
      return static_cast<T*>(this);
    }
    return nullptr;
  }
  //! Type safe cast.
  template <typename T>
  const T* safe_as() const {
    if (std::strcmp(type_info(), T::__type_info__) == 0) {
      return static_cast<const T*>(this);
    }
    return nullptr;
  }

  //! Check if the type is right.
  template <typename T>
  bool is_type() const {
    if (std::strcmp(type_info(), T::__type_info__) == 0) {
      return true;
    }
    return false;
  }

  //! The reference count, which make all the derived type able to share.
  mutable RefCount __ref_count__;
};

using object_ptr = Object*;
using shared_object = Shared<Object>;

}  // namespace common
}  // namespace infrt

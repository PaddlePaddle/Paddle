// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/common/shared.h"

namespace cinn {
namespace common {

template <typename T>
class Shared;
/**
 * Object is the basic element in the CINN, with `Shared` wrapper, the object
 * can be shared across the system.
 */
struct Object {
  //! Get the type representation of this object.
  virtual const char* type_info() const = 0;

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

/*
 * \brief Delete the default copy/move constructor and assign operator
 * \param TypeName The class typename.
 */
#define CINN_DELETE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_NODE(TypeName) \
  TypeName(const TypeName& other) = delete;                        \
  TypeName(TypeName&& other) = delete;                             \
  TypeName& operator=(const TypeName& other) = delete;             \
  TypeName& operator=(TypeName&& other) = delete;

/*
 * \brief Define the default copy/move constructor and assign operator
 * \param TypeName The class typename.
 */
#define CINN_DEFINE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_REF(TypeName) \
  TypeName(const TypeName& other) = default;                      \
  TypeName(TypeName&& other) = default;                           \
  TypeName& operator=(const TypeName& other) = default;           \
  TypeName& operator=(TypeName&& other) = default;

/*
 * \brief Define object reference methods
 * \param TypeName The object type name
 * \param ParentType The parent type of the objectref
 * \param ObjectName The type name of the object
 */
#define CINN_DEFINE_OBJECT_REF_METHODS(TypeName, ParentType, ObjectName) \
  TypeName() = default;                                                  \
  explicit TypeName(ObjectName* n) : ParentType(n) {}                    \
  explicit TypeName(const cinn::common::Shared<Object>& ref)             \
      : ParentType(ref) {}                                               \
  CINN_DEFINE_COPY_MOVE_AND_ASSIGN_FOR_OBJECT_REF(TypeName);             \
  const ObjectName* operator->() const {                                 \
    return static_cast<const ObjectName*>(this->p_);                     \
  }                                                                      \
  ObjectName* operator->() { return static_cast<ObjectName*>(this->p_); }

}  // namespace common
}  // namespace cinn

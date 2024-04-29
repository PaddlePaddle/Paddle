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

#include <functional>

#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/utils/test_macros.h"

namespace pir {

///
/// \brief TypeId is the unique identification of Type, each Type corresponds to
/// a unique TypeId, the same id indicates the same Type class. TypeId provides
/// an instantiation interface: TypeId::get.
///
/// Example:
/// \code{cpp}
///   class TypeA {};
///   TypeId type_a_id = TypeId::get<TypeA>();
/// \endcode
///
class TypeId {
  struct Storage {};

 public:
  ///
  /// \brief Returns the unique TypeId of Type T.
  ///
  /// \return The unique TypeId of Type T.
  ///
  template <typename T>
  static TypeId get();

  TypeId() = default;

  TypeId(const TypeId &other) = default;

  TypeId &operator=(const TypeId &other) = default;

  ///
  /// \brief Support PointerLikeTypeTraits.
  ///
  operator void *() const { return storage_; }
  static TypeId RecoverFromVoidPointer(void *pointer) {
    return TypeId(static_cast<Storage *>(pointer));
  }

  ///
  /// \brief Comparison operations.
  ///
  inline bool operator==(const TypeId &other) const {
    return storage_ == other.storage_;
  }
  inline bool operator!=(const TypeId &other) const {
    return !(*this == other);
  }
  inline bool operator<(const TypeId &other) const {
    return storage_ < other.storage_;
  }

 private:
  ///
  /// \brief Construct a TypeId and initialize storage.
  ///
  /// \param storage The storage of this TypeId.
  ///
  explicit TypeId(Storage *storage) : storage_(storage) {}

  Storage *storage_{nullptr};
};

namespace detail {
class alignas(8) UniqueingId {
 public:
  UniqueingId() = default;
  UniqueingId(const UniqueingId &) = delete;
  UniqueingId &operator=(const UniqueingId &) = delete;
  UniqueingId(UniqueingId &&) = delete;
  UniqueingId &operator=(UniqueingId &&) = delete;

  operator TypeId() { return id(); }
  TypeId id() { return TypeId::RecoverFromVoidPointer(this); }
};

template <typename T>
class TypeIdResolver;

}  // namespace detail

template <typename T>
TypeId TypeId::get() {
  return detail::TypeIdResolver<T>::Resolve();
}

#define IR_DECLARE_EXPLICIT_TYPE_ID(TYPE_CLASS) \
  namespace pir {                               \
  namespace detail {                            \
  template <>                                   \
  class TEST_API TypeIdResolver<TYPE_CLASS> {   \
   public:                                      \
    static TypeId Resolve() { return id_; }     \
    static UniqueingId id_;                     \
  };                                            \
  }                                             \
  }  // namespace pir

#define IR_EXPORT_DECLARE_EXPLICIT_TYPE_ID(TYPE_CLASS) \
  namespace pir {                                      \
  namespace detail {                                   \
  template <>                                          \
  class IR_API TypeIdResolver<TYPE_CLASS> {            \
   public:                                             \
    static TypeId Resolve() { return id_; }            \
    static UniqueingId id_;                            \
  };                                                   \
  }                                                    \
  }  // namespace pir

#define IR_DEFINE_EXPLICIT_TYPE_ID(TYPE_CLASS)      \
  namespace pir {                                   \
  namespace detail {                                \
  UniqueingId TypeIdResolver<TYPE_CLASS>::id_ = {}; \
  }                                                 \
  }  // namespace pir

}  // namespace pir

namespace std {
///
/// \brief Enable hashing TypeId instances.
///
template <>
struct hash<pir::TypeId> {
  std::size_t operator()(const pir::TypeId &obj) const {
    return std::hash<void *>()(obj);
  }
};
}  // namespace std

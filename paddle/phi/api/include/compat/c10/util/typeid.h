// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Half.h>
#include <c10/util/TypeIndex.h>

#include <c10/core/ScalarType.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "paddle/common/enforce.h"

// ---------------------------------------------------------------------------
// Auxiliary macros not provided by the compat Macros.h
// ---------------------------------------------------------------------------

#ifndef C10_LIKELY
#ifdef _MSC_VER
#define C10_LIKELY(val) (val)
#else
#define C10_LIKELY(val) (__builtin_expect(static_cast<bool>(val), 1))
#endif
#endif
#ifndef C10_UNLIKELY
#ifdef _MSC_VER
#define C10_UNLIKELY(val) (val)
#else
#define C10_UNLIKELY(val) (__builtin_expect(static_cast<bool>(val), 0))
#endif
#endif
#ifndef C10_API
#define C10_API PADDLE_API
#endif
#ifndef C10_EXPORT
#ifdef _MSC_VER
#define C10_EXPORT __declspec(dllexport)
#else
#define C10_EXPORT __attribute__((visibility("default")))
#endif
#endif
#ifndef C10_ALWAYS_INLINE
#ifdef _MSC_VER
#define C10_ALWAYS_INLINE __forceinline
#else
#define C10_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif
#endif
#ifndef TORCH_INTERNAL_ASSERT_DEBUG_ONLY
#ifdef NDEBUG
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) ((void)0)
#else
#define TORCH_INTERNAL_ASSERT_DEBUG_ONLY(...) TORCH_INTERNAL_ASSERT(__VA_ARGS__)
#endif
#endif

// ---------------------------------------------------------------------------
// caffe2::TypeIdentifier
// ---------------------------------------------------------------------------

namespace caffe2 {

/**
 * A unique run-time type identifier.
 */
class C10_API TypeIdentifier final {
 public:
  friend std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId);
  friend bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) noexcept;

  template <typename T>
  static constexpr TypeIdentifier Get() noexcept {
    return TypeIdentifier(c10::util::get_type_index<T>());
  }

  static constexpr TypeIdentifier uninitialized() noexcept {
    return TypeIdentifier(c10::util::type_index{0});
  }

  uint64_t underlyingId() const noexcept { return id_.underlyingId(); }

  bool operator==(TypeIdentifier other) const noexcept {
    return id_ == other.id_;
  }
  bool operator!=(TypeIdentifier other) const noexcept {
    return id_ != other.id_;
  }

 private:
  constexpr explicit TypeIdentifier(c10::util::type_index id) noexcept
      : id_(id) {}
  c10::util::type_index id_;
};

inline bool operator<(TypeIdentifier lhs, TypeIdentifier rhs) noexcept {
  return lhs.id_ < rhs.id_;
}
inline std::ostream& operator<<(std::ostream& stream, TypeIdentifier typeId) {
  return stream << typeId.underlyingId();
}

}  // namespace caffe2

// ---------------------------------------------------------------------------
// at::DataType alias
// ---------------------------------------------------------------------------

namespace at {
using DataType = caffe2::TypeIdentifier;
}  // namespace at

// ---------------------------------------------------------------------------
// std::hash specialisation so TypeIdentifier can be used in unordered maps
// ---------------------------------------------------------------------------

namespace std {
template <>
struct hash<caffe2::TypeIdentifier> {
  std::size_t operator()(caffe2::TypeIdentifier id) const noexcept {
    return id.underlyingId();
  }
};
}  // namespace std

// ---------------------------------------------------------------------------
// caffe2::detail  –  TypeMetaData + helper templates
// ---------------------------------------------------------------------------

namespace caffe2 {
namespace detail {

/**
 * Per-type metadata record.  One instance lives per registered type.
 */
struct TypeMetaData final {
  using New = void*();
  using PlacementNew = void(void*, size_t);
  using Copy = void(const void*, void*, size_t);
  using PlacementDelete = void(void*, size_t);
  using Delete = void(void*);

  constexpr TypeMetaData() noexcept
      : itemsize_(0),
        new_(nullptr),
        placementNew_(nullptr),
        copy_(nullptr),
        placementDelete_(nullptr),
        delete_(nullptr),
        id_(TypeIdentifier::uninitialized()),
        name_("nullptr (uninitialized)") {}

  constexpr TypeMetaData(size_t itemsize,
                         New* newFn,
                         PlacementNew* placementNew,
                         Copy* copy,
                         PlacementDelete* placementDelete,
                         Delete* deleteFn,
                         TypeIdentifier id,
                         std::string_view name) noexcept
      : itemsize_(itemsize),
        new_(newFn),
        placementNew_(placementNew),
        copy_(copy),
        placementDelete_(placementDelete),
        delete_(deleteFn),
        id_(id),
        name_(name) {}

  size_t itemsize_;
  New* new_;
  PlacementNew* placementNew_;
  Copy* copy_;
  PlacementDelete* placementDelete_;
  Delete* delete_;
  TypeIdentifier id_;
  std::string_view name_;
};

// Error helper – keeps this header free of heavy includes.
[[noreturn]] inline void _ThrowRuntimeTypeLogicError(const std::string& msg) {
  PADDLE_THROW(common::errors::InvalidArgument(msg));
}

// ---------------------------------------------------------------------------
// Trait: treat reduced-precision scalars as "fundamental" (skip ctor/dtor).
// ---------------------------------------------------------------------------
template <typename T>
struct is_paddle_fundamental : std::is_fundamental<T> {};
template <>
struct is_paddle_fundamental<at::Half> : std::true_type {};
template <>
struct is_paddle_fundamental<at::BFloat16> : std::true_type {};
template <>
struct is_paddle_fundamental<c10::Float8_e4m3fn> : std::true_type {};
template <>
struct is_paddle_fundamental<c10::Float8_e5m2> : std::true_type {};

// ---------------------------------------------------------------------------
// PlacementNew helpers
// ---------------------------------------------------------------------------
template <typename T>
inline void _PlacementNew(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) new (typed_ptr + i) T;
}

template <typename T>
inline void _PlacementNewNotDefault(void* /*ptr*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(std::string("Type ") + typeid(T).name() +
                              " is not default-constructible.");
}

template <typename T,
          std::enable_if_t<std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return (is_paddle_fundamental<T>::value || std::is_pointer_v<T>)
             ? nullptr
             : &_PlacementNew<T>;
}

template <typename T,
          std::enable_if_t<!std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::PlacementNew* _PickPlacementNew() {
  return &_PlacementNewNotDefault<T>;
}

// ---------------------------------------------------------------------------
// New helpers
// ---------------------------------------------------------------------------
template <typename T>
inline void* _New() {
  return new T;
}

template <typename T>
inline void* _NewNotDefault() {
  _ThrowRuntimeTypeLogicError(std::string("Type ") + typeid(T).name() +
                              " is not default-constructible.");
}

template <typename T,
          std::enable_if_t<std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_New<T>;
}

template <typename T,
          std::enable_if_t<!std::is_default_constructible_v<T>>* = nullptr>
inline constexpr TypeMetaData::New* _PickNew() {
  return &_NewNotDefault<T>;
}

// ---------------------------------------------------------------------------
// Copy helpers
// ---------------------------------------------------------------------------
template <typename T>
inline void _Copy(const void* src, void* dst, size_t n) {
  const T* typed_src = static_cast<const T*>(src);
  T* typed_dst = static_cast<T*>(dst);
  for (size_t i = 0; i < n; ++i) typed_dst[i] = typed_src[i];
}

template <typename T>
inline void _CopyNotAllowed(const void* /*src*/, void* /*dst*/, size_t /*n*/) {
  _ThrowRuntimeTypeLogicError(std::string("Type ") + typeid(T).name() +
                              " does not allow assignment.");
}

template <typename T, std::enable_if_t<std::is_copy_assignable_v<T>>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return (is_paddle_fundamental<T>::value || std::is_pointer_v<T>) ? nullptr
                                                                   : &_Copy<T>;
}

template <typename T,
          std::enable_if_t<!std::is_copy_assignable_v<T>>* = nullptr>
inline constexpr TypeMetaData::Copy* _PickCopy() {
  return &_CopyNotAllowed<T>;
}

// ---------------------------------------------------------------------------
// PlacementDelete helpers
// ---------------------------------------------------------------------------
template <typename T>
inline void _PlacementDelete(void* ptr, size_t n) {
  T* typed_ptr = static_cast<T*>(ptr);
  for (size_t i = 0; i < n; ++i) typed_ptr[i].~T();
}

template <typename T>
inline constexpr TypeMetaData::PlacementDelete* _PickPlacementDelete() {
  return (is_paddle_fundamental<T>::value || std::is_pointer_v<T>)
             ? nullptr
             : &_PlacementDelete<T>;
}

// ---------------------------------------------------------------------------
// Delete helpers
// ---------------------------------------------------------------------------
template <typename T>
inline void _Delete(void* ptr) {
  delete static_cast<T*>(ptr);
}

template <class T>
inline constexpr TypeMetaData::Delete* _PickDelete() noexcept {
  return &_Delete<T>;
}

// Sentinel type for uninitialized TypeMeta.
class _Uninitialized final {};

}  // namespace detail

// ---------------------------------------------------------------------------
// caffe2::TypeMeta
// ---------------------------------------------------------------------------

/**
 * TypeMeta is a thin class that stores the type of a container (e.g. Blob or
 * Tensor elements) with a unique run-time id.  It mirrors the PyTorch / Caffe2
 * TypeMeta API so that code written against libtorch's typeid.h compiles
 * against this compat header without changes.
 */
class C10_API TypeMeta final {
 public:
  using New = detail::TypeMetaData::New;
  using PlacementNew = detail::TypeMetaData::PlacementNew;
  using Copy = detail::TypeMetaData::Copy;
  using PlacementDelete = detail::TypeMetaData::PlacementDelete;
  using Delete = detail::TypeMetaData::Delete;

  // Default-constructs to "Undefined / uninitialized".
  // NOTE: body is defined AFTER the _Uninitialized specialization below
  // to avoid "specialization after instantiation" errors.
  TypeMeta() noexcept;
  ~TypeMeta() = default;

  TypeMeta(const TypeMeta& src) noexcept = default;
  TypeMeta& operator=(const TypeMeta& src) noexcept = default;
  TypeMeta(TypeMeta&& src) noexcept = default;
  TypeMeta& operator=(TypeMeta&& src) noexcept = default;

  inline TypeMeta& operator=(c10::ScalarType scalar_type) noexcept {
    index_ = static_cast<uint16_t>(scalar_type);
    return *this;
  }

  // ------------------------------------------------------------------
  // Accessors
  // ------------------------------------------------------------------
  TypeIdentifier id() const noexcept { return data().id_; }

  inline bool isScalarType() const noexcept {
    return index_ <= static_cast<uint16_t>(c10::ScalarType::Undefined);
  }
  inline bool isScalarType(c10::ScalarType scalar_type) const noexcept {
    return index_ == static_cast<uint16_t>(scalar_type);
  }

  inline size_t itemsize() const noexcept { return data().itemsize_; }

  New* newFn() const noexcept { return data().new_; }
  PlacementNew* placementNew() const noexcept { return data().placementNew_; }
  Copy* copy() const noexcept { return data().copy_; }
  PlacementDelete* placementDelete() const noexcept {
    return data().placementDelete_;
  }
  Delete* deleteFn() const noexcept { return data().delete_; }
  std::string_view name() const noexcept { return data().name_; }

  friend bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept;

  template <typename T>
  bool Match() const noexcept {
    return *this == Make<T>();
  }

  // ------------------------------------------------------------------
  // Static helpers
  // ------------------------------------------------------------------
  template <class T>
  static constexpr TypeIdentifier Id() noexcept {
    return TypeIdentifier::Get<T>();
  }

  template <class T>
  static std::string_view TypeName() noexcept {
    return typeid(T).name();
  }

  template <class T>
  static constexpr size_t ItemSize() noexcept {
    return sizeof(T);
  }

  /** Returns a TypeMeta for type T. */
  template <typename T>
  static TypeMeta Make() {
    return TypeMeta(_typeMetaData<T>());
  }

  /** Convert ScalarType enum → TypeMeta. */
  static inline TypeMeta fromScalarType(c10::ScalarType scalar_type) {
    const auto index = static_cast<uint16_t>(scalar_type);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        index <= static_cast<uint16_t>(c10::ScalarType::Undefined),
        "Unrecognized ScalarType");
    return TypeMeta(index);
  }

  /** Convert TypeMeta → ScalarType enum. */
  inline c10::ScalarType toScalarType() const {
    if (C10_LIKELY(isScalarType())) {
      return static_cast<c10::ScalarType>(index_);
    }
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported TypeMeta in toScalarType (index=%d)",
        static_cast<int>(index_)));
  }

  // ------------------------------------------------------------------
  // Dynamic type registration (mirrors CAFFE_KNOWN_TYPE machinery)
  // ------------------------------------------------------------------
  template <class T>
  static uint16_t addTypeMetaData() {
    const auto identifier = TypeIdentifier::Get<T>();
    std::lock_guard<std::mutex> lock(getTypeMetaDatasLock());
    // Check whether already registered (e.g. from another DSO).
    const uint16_t existing = existingMetaDataIndexForType(identifier);
    if (existing != kMaxTypeIndex) return existing;
    const uint16_t index = nextTypeIndex++;
    TORCH_CHECK(index <= kMaxTypeIndex,
                "Maximum number of CAFFE_KNOWN_TYPE declarations exceeded.");
    typeMetaDatas()[index] =
        detail::TypeMetaData{sizeof(T),
                             detail::_PickNew<T>(),
                             detail::_PickPlacementNew<T>(),
                             detail::_PickCopy<T>(),
                             detail::_PickPlacementDelete<T>(),
                             detail::_PickDelete<T>(),
                             identifier,
                             typeid(T).name()};
    return index;
  }

 private:
  explicit TypeMeta(uint16_t index) noexcept : index_(index) {}

  // Maximum number of type-metadata slots (scalar + custom).
  static constexpr uint16_t kMaxTypeIndex = 255;

  static std::mutex& getTypeMetaDatasLock();

  static uint16_t nextTypeIndex;

  static detail::TypeMetaData* typeMetaDatas();

  static uint16_t existingMetaDataIndexForType(TypeIdentifier identifier);

  // Template specialisations return indexes into typeMetaDatas().
  // Defined below the class for scalar types; compiled-in for custom types.
  template <class T>
  static uint16_t _typeMetaData() noexcept;

  uint16_t index_;

  inline const detail::TypeMetaData& data() const {
    return typeMetaDatas()[index_];
  }
};

// ---------------------------------------------------------------------------
// Specialisations of TypeMeta::_typeMetaData for ScalarType types
// ---------------------------------------------------------------------------

// 3-argument AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS
#define DEFINE_SCALAR_METADATA_INSTANCE(T, _2, name)                \
  template <>                                                       \
  inline constexpr uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    return static_cast<uint16_t>(c10::ScalarType::name);            \
  }
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_METADATA_INSTANCE)
#undef DEFINE_SCALAR_METADATA_INSTANCE

// _Uninitialized → Undefined slot
template <>
inline constexpr uint16_t
TypeMeta::_typeMetaData<detail::_Uninitialized>() noexcept {
  return static_cast<uint16_t>(c10::ScalarType::Undefined);
}

// Default constructor defined here so that _typeMetaData<_Uninitialized> is
// already specialised before it is first called.
inline TypeMeta::TypeMeta() noexcept
    : index_(_typeMetaData<detail::_Uninitialized>()) {}

// ---------------------------------------------------------------------------
// Comparison / stream operators
// ---------------------------------------------------------------------------

inline bool operator==(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return lhs.index_ == rhs.index_;
}
inline bool operator!=(const TypeMeta& lhs, const TypeMeta& rhs) noexcept {
  return !(lhs == rhs);
}
inline std::ostream& operator<<(std::ostream& stream,
                                caffe2::TypeMeta typeMeta) {
  return stream << typeMeta.name();
}

// ---------------------------------------------------------------------------
// CAFFE_KNOWN_TYPE macros (compat versions)
// ---------------------------------------------------------------------------

#if defined(_MSC_VER) || defined(__clang__)
#define EXPORT_IF_NOT_GCC C10_EXPORT
#else
#define EXPORT_IF_NOT_GCC
#endif

// For use in a .cpp file.
#define CAFFE_KNOWN_TYPE(T)                                          \
  template uint16_t TypeMeta::addTypeMetaData<T>();                  \
  template <>                                                        \
  EXPORT_IF_NOT_GCC uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    static const uint16_t index = addTypeMetaData<T>();              \
    return index;                                                    \
  }

// For use in a .cpp file when a declaration in the header is provided.
#define CAFFE_DEFINE_KNOWN_TYPE(T, ident)                          \
  template uint16_t TypeMeta::addTypeMetaData<T>();                \
  namespace detail {                                               \
  EXPORT_IF_NOT_GCC extern const uint16_t ident##_metadata_index = \
      TypeMeta::addTypeMetaData<T>();                              \
  } /* namespace detail */

// Declaration counterpart: provides an inline fast-path via a detail var.
// NOTE: On MSVC, directly referencing cross-DLL const data symbols is fragile
// and can cause unresolved externals during libpaddle linking. Use a
// function-local static cache there and keep non-MSVC behavior aligned with
// upstream declare/define model.
#if defined(_MSC_VER)
#define CAFFE_DECLARE_KNOWN_TYPE(T, ident)                           \
  extern template uint16_t TypeMeta::addTypeMetaData<T>();           \
  namespace detail {                                                 \
  extern C10_API const uint16_t ident##_metadata_index;              \
  } /* namespace detail */                                           \
  template <>                                                        \
  C10_ALWAYS_INLINE uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    static const uint16_t index = addTypeMetaData<T>();              \
    return index;                                                    \
  }
#else
#define CAFFE_DECLARE_KNOWN_TYPE(T, ident)                 \
  extern template uint16_t TypeMeta::addTypeMetaData<T>(); \
  namespace detail {                                       \
  extern C10_API const uint16_t ident##_metadata_index;    \
  } /* namespace detail */                                 \
  template <>                                              \
  EXPORT_IF_NOT_GCC C10_ALWAYS_INLINE uint16_t             \
  TypeMeta::_typeMetaData<T>() noexcept {                  \
    return detail::ident##_metadata_index;                 \
  }
#endif

// Header-safe variant: lazy static, no external .cpp needed.
#define CAFFE_KNOWN_TYPE_NOEXPORT(T)                      \
  template <>                                             \
  inline uint16_t TypeMeta::_typeMetaData<T>() noexcept { \
    static const uint16_t index = addTypeMetaData<T>();   \
    return index;                                         \
  }

// ---------------------------------------------------------------------------
// Built-in known types
// ---------------------------------------------------------------------------

namespace detail {
template <class T>
class _guard_long_unique_dummy final {};

template <class T>
using _guard_long_unique =
    std::conditional_t<std::is_same_v<long, int32_t> ||  // NOLINT(runtime/int)
                           std::is_same_v<long,          // NOLINT(runtime/int)
                                          int64_t>,
                       _guard_long_unique_dummy<T>,
                       T>;
}  // namespace detail

CAFFE_DECLARE_KNOWN_TYPE(std::string, std_string)
CAFFE_DECLARE_KNOWN_TYPE(char, char)
CAFFE_DECLARE_KNOWN_TYPE(std::unique_ptr<std::mutex>, std_unique_ptr_std_mutex)
CAFFE_DECLARE_KNOWN_TYPE(std::unique_ptr<std::atomic<bool>>,
                         std_unique_ptr_std_atomic_bool)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<int32_t>, std_vector_int32_t)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<int64_t>, std_vector_int64_t)
CAFFE_DECLARE_KNOWN_TYPE(std::vector<unsigned long>,  // NOLINT(runtime/int)
                         std_vector_unsigned_long)
CAFFE_DECLARE_KNOWN_TYPE(bool*, bool_ptr)
CAFFE_DECLARE_KNOWN_TYPE(char*, char_ptr)
CAFFE_DECLARE_KNOWN_TYPE(int*, int_ptr)
CAFFE_DECLARE_KNOWN_TYPE(
    detail::_guard_long_unique<long>,  // NOLINT(runtime/int)
    detail_guard_long_unique_long)
CAFFE_DECLARE_KNOWN_TYPE(
    detail::_guard_long_unique<std::vector<long>>,  // NOLINT(runtime/int)
    detail_guard_long_unique_std_vector_long)
CAFFE_DECLARE_KNOWN_TYPE(float*, float_ptr)
CAFFE_DECLARE_KNOWN_TYPE(at::Half*, at_Half)

}  // namespace caffe2

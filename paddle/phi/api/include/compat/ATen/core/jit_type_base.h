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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ATen/core/type_ptr.h"
#include "c10/util/ArrayRef.h"
#include "c10/util/Exception.h"

namespace c10 {

#define C10_FORALL_TYPES(_) \
  _(TensorType)             \
  _(StringType)             \
  _(IntType)                \
  _(FloatType)              \
  _(BoolType)               \
  _(NoneType)               \
  _(TupleType)              \
  _(NumberType)             \
  _(OptionalType)           \
  _(UnionType)              \
  _(DeviceObjType)          \
  _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
  C10_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

struct Type;
struct SharedType;
using TypePrinter = std::function<std::optional<std::string>(const Type&)>;

namespace detail {
template <typename T>
struct IsSingletonType : std::false_type {};
}  // namespace detail
#define TORCH_DECLARE_SINGLETON(Type)               \
  struct Type;                                      \
  namespace detail {                                \
  template <>                                       \
  struct IsSingletonType<Type> : std::true_type {}; \
  }

TORCH_DECLARE_SINGLETON(NumberType)
TORCH_DECLARE_SINGLETON(TensorType)
TORCH_DECLARE_SINGLETON(StringType)
TORCH_DECLARE_SINGLETON(IntType)
TORCH_DECLARE_SINGLETON(FloatType)
TORCH_DECLARE_SINGLETON(BoolType)
TORCH_DECLARE_SINGLETON(NoneType)
TORCH_DECLARE_SINGLETON(TupleType)
TORCH_DECLARE_SINGLETON(OptionalType)
TORCH_DECLARE_SINGLETON(DeviceObjType)

namespace detail {
template <typename T, typename Enable = void>
struct CastReturnType {
  using type = std::shared_ptr<T>;
};

template <typename T>
struct CastReturnType<T, std::enable_if_t<IsSingletonType<T>::value>> {
  using type = SingletonTypePtr<T>;
};

template <typename T, typename Enable = void>
struct CastConstReturnType {
  using type = std::shared_ptr<const T>;
};

template <typename T>
struct CastConstReturnType<T, std::enable_if_t<IsSingletonType<T>::value>> {
  using type = SingletonTypePtr<const T>;
};

}  // namespace detail

struct PADDLE_API Type {
  friend PADDLE_API bool operator==(const Type& lhs, const Type& rhs);

 private:
  TypeKind kind_;

 protected:
  explicit Type(TypeKind kind) : kind_(kind) {}

  Type(const Type&) = default;
  Type& operator=(const Type&) = default;
  Type(Type&&) noexcept = default;
  Type& operator=(Type&&) noexcept = default;

  virtual std::string annotation_str_impl(
      const TypePrinter& /*printer*/) const {
    return str();
  }
  virtual bool equals(const Type& rhs) const = 0;
  virtual bool symmetric() const { return true; }

 public:
  template <typename T>
  class SingletonOrSharedTypePtr {
   public:
    using element_type = typename std::shared_ptr<T>::element_type;

    SingletonOrSharedTypePtr() = default;

    SingletonOrSharedTypePtr(std::shared_ptr<T> x)  // NOLINT(runtime/explicit)
        : repr_(std::move(x)) {}

    template <typename U,
              std::enable_if_t<std::is_convertible_v<U*, T*>, bool> = true>
    SingletonOrSharedTypePtr(std::shared_ptr<U> x)  // NOLINT(runtime/explicit)
        : repr_(std::move(x)) {}

    SingletonOrSharedTypePtr(std::nullptr_t)  // NOLINT(runtime/explicit)
        : repr_(nullptr) {}

    SingletonOrSharedTypePtr(SingletonTypePtr<T> p)  // NOLINT(runtime/explicit)
        : repr_(makeSingletonSharedPtr(p.get())) {}

    template <typename U,
              std::enable_if_t<std::is_convertible_v<U*, T*>, bool> = true>
    SingletonOrSharedTypePtr(SingletonTypePtr<U> p)  // NOLINT(runtime/explicit)
        : repr_(makeSingletonSharedPtr(static_cast<T*>(p.get()))) {}

    SingletonOrSharedTypePtr(const SingletonOrSharedTypePtr&) = default;
    SingletonOrSharedTypePtr(SingletonOrSharedTypePtr&&) noexcept = default;
    SingletonOrSharedTypePtr& operator=(const SingletonOrSharedTypePtr&) =
        default;
    SingletonOrSharedTypePtr& operator=(SingletonOrSharedTypePtr&&) noexcept =
        default;
    ~SingletonOrSharedTypePtr() = default;

    T* get() const { return repr_.get(); }

    operator bool() const { return repr_ != nullptr; }

    bool operator==(std::nullptr_t) const { return repr_ == nullptr; }

    bool operator!=(std::nullptr_t) const { return repr_ != nullptr; }

    template <typename U = T,
              std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>,
                               bool> = true>
    U& operator*() const {
      return *get();
    }

    T* operator->() const { return get(); }

   private:
    static std::shared_ptr<T> makeSingletonSharedPtr(T* ptr) {
      return std::shared_ptr<T>(std::shared_ptr<T>(), ptr);
    }

    std::shared_ptr<T> repr_;
  };

  using TypePtr = SingletonOrSharedTypePtr<Type>;

  virtual bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const;
  bool isSubtypeOf(const Type& rhs) const {
    return isSubtypeOfExt(rhs, nullptr);
  }

  // Compatibility shims to accommodate existing code that passes shared_ptrs
  // around. Ideally, we would just delete this, but it should be harmless.
  template <typename T>
  std::enable_if_t<std::is_base_of_v<Type, T>, bool> isSubtypeOf(
      const std::shared_ptr<T>& rhs) const {
    return isSubtypeOf(*rhs);
  }

  virtual std::string str() const = 0;

  std::string annotation_str(const TypePrinter& printer) const {
    if (printer) {
      if (auto renamed = printer(*this)) {
        return *renamed;
      }
    }
    return annotation_str_impl(printer);
  }
  std::string annotation_str() const { return annotation_str(nullptr); }

  virtual std::string repr_str() const { return annotation_str(); }

  TypeKind kind() const { return kind_; }

  template <typename T,
            std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (auto* typed = dynamic_cast<T*>(this)) {
      return std::static_pointer_cast<T>(typed->shared_from_this());
    }
    return nullptr;
  }
  template <typename T,
            std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastReturnType<T>::type cast() {
    if (auto* typed = dynamic_cast<T*>(this)) {
      return typename detail::CastReturnType<T>::type(typed);
    }
    return nullptr;
  }
  template <typename T,
            std::enable_if_t<!detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (auto* typed = dynamic_cast<const T*>(this)) {
      return std::static_pointer_cast<const T>(typed->shared_from_this());
    }
    return nullptr;
  }
  template <typename T,
            std::enable_if_t<detail::IsSingletonType<T>::value, bool> = true>
  typename detail::CastConstReturnType<T>::type cast() const {
    if (auto* typed = dynamic_cast<const T*>(this)) {
      return typename detail::CastConstReturnType<T>::type(typed);
    }
    return nullptr;
  }
  virtual ~Type() = default;
  virtual at::ArrayRef<TypePtr> containedTypes() const { return {}; }
  virtual TypePtr createWithContained(
      std::vector<TypePtr> /*contained_types*/) const {
    TORCH_CHECK(
        false,
        "type with contained types did not overload createWithContained: ",
        str());
  }
};

template <typename T>
using SingletonOrSharedTypePtr = Type::SingletonOrSharedTypePtr<T>;

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x,
                const SingletonOrSharedTypePtr<U>& y) {
  return static_cast<const void*>(x.get()) == static_cast<const void*>(y.get());
}

template <typename T, typename U>
bool operator==(const SingletonOrSharedTypePtr<T>& x,
                const SingletonTypePtr<U>& y) {
  return static_cast<const void*>(x.get()) == static_cast<const void*>(y.get());
}

template <typename T, typename U>
bool operator==(const SingletonTypePtr<T>& x,
                const SingletonOrSharedTypePtr<U>& y) {
  return static_cast<const void*>(x.get()) == static_cast<const void*>(y.get());
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x,
                const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonOrSharedTypePtr<T>& x,
                const SingletonTypePtr<U>& y) {
  return !(x == y);
}

template <typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& x,
                const SingletonOrSharedTypePtr<U>& y) {
  return !(x == y);
}

using TypePtr = SingletonOrSharedTypePtr<Type>;

// Base class for Types that are guaranteed to be owned by std::shared_ptr.
struct PADDLE_API SharedType : public Type,
                               public std::enable_shared_from_this<SharedType> {
  using Type::Type;
};

inline bool operator==(const Type& lhs, const Type& rhs) {
  if (!rhs.symmetric()) {
    return rhs.equals(lhs);
  }
  return lhs.equals(rhs);
}

inline bool Type::isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const {
  if (*this == rhs) {
    return true;
  }
  if (rhs.kind() == TypeKind::OptionalType) {
    for (const auto& inner : rhs.containedTypes()) {
      if (*this == *inner) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace c10

namespace std {
template <typename T>
struct hash<c10::SingletonOrSharedTypePtr<T>> {
  size_t operator()(const c10::SingletonOrSharedTypePtr<T>& x) const {
    return std::hash<T*>()(x.get());
  }
};
}  // namespace std

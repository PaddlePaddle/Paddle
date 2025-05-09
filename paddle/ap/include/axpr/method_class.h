// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef _WIN32
#include <experimental/type_traits>
#endif
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include "paddle/ap/include/axpr/adt.h"
#include "paddle/ap/include/axpr/builtin_class_instance.h"
#include "paddle/ap/include/axpr/builtin_func_type.h"
#include "paddle/ap/include/axpr/class_instance.h"
#include "paddle/ap/include/axpr/constants.h"
#include "paddle/ap/include/axpr/type.h"

namespace ap::axpr {

template <typename ValueT>
class InterpreterBase;

template <typename ValueT>
using BuiltinUnaryFuncImpl =
    std::variant<adt::Nothing,
                 adt::Result<ValueT> (*)(const ValueT&),
                 adt::Result<ValueT> (*)(InterpreterBase<ValueT>*,
                                         const ValueT&)>;

template <typename ValueT>
struct BuiltinUnaryFunc : public BuiltinUnaryFuncImpl<ValueT> {
  using BuiltinUnaryFuncImpl<ValueT>::BuiltinUnaryFuncImpl;
  ADT_DEFINE_VARIANT_METHODS(BuiltinUnaryFuncImpl<ValueT>);
};

template <typename ValueT, BuiltinFuncType<ValueT> BuiltinFunc>
adt::Result<ValueT> UnaryFuncReturnCapturedValue(const ValueT&) {
  return BuiltinFunc;
}

template <typename ValueT>
using BuiltinBinaryFuncImpl =
    std::variant<adt::Nothing,
                 adt::Result<ValueT> (*)(const ValueT&, const ValueT&),
                 adt::Result<ValueT> (*)(
                     InterpreterBase<ValueT>*, const ValueT&, const ValueT&)>;

template <typename ValueT>
struct BuiltinBinaryFunc : public BuiltinBinaryFuncImpl<ValueT> {
  using BuiltinBinaryFuncImpl<ValueT>::BuiltinBinaryFuncImpl;
  ADT_DEFINE_VARIANT_METHODS(BuiltinBinaryFuncImpl<ValueT>);
};

template <typename ValueT>
struct EmptyMethodClass {
  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    return adt::Nothing{};
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    return adt::Nothing{};
  }
};

template <typename ValueT, typename T>
struct MethodClassImpl;

namespace detail {

#ifndef _WIN32
template <template <typename...> class Op, typename... Args>
constexpr bool is_detected_v = std::experimental::is_detected_v<Op, Args...>;
#else
template <template <typename...> class Op, typename, typename... Args>
struct detector : std::false_type {};

template <template <typename...> class Op, typename... Args>
struct detector<Op, std::void_t<Op<Args...>>, Args...> : std::true_type {};

template <template <typename...> class Op, typename... Args>
constexpr bool is_detected_v = detector<Op, void, Args...>::value;
#endif

template <typename ValueT, typename T, typename BuiltinSymbol>
struct BuiltinMethodHelperImpl;

#define SPECIALIZE_BuiltinMethodHelperImpl(symbol_name, op)                  \
  template <typename ValueT, typename T>                                     \
  struct BuiltinMethodHelperImpl<ValueT, T, builtin_symbol::symbol_name> {   \
    using This =                                                             \
        BuiltinMethodHelperImpl<ValueT, T, builtin_symbol::symbol_name>;     \
                                                                             \
    template <typename ObjT>                                                 \
    using UnaryMethodRetT =                                                  \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<const ObjT&>()));                                   \
                                                                             \
    template <typename ObjT>                                                 \
    using HighOrderUnaryMethodRetT =                                         \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<InterpreterBase<ValueT>*>(),                        \
            std::declval<const ObjT&>()));                                   \
                                                                             \
    static constexpr bool HasUnaryMethod() {                                 \
      return builtin_symbol::symbol_name::num_operands == 1 &&               \
             is_detected_v<UnaryMethodRetT, T>;                              \
    }                                                                        \
                                                                             \
    static constexpr bool HasHighOrderUnaryMethod() {                        \
      return builtin_symbol::symbol_name::num_operands == 1 &&               \
             is_detected_v<HighOrderUnaryMethodRetT, T>;                     \
    }                                                                        \
                                                                             \
    static adt::Result<ValueT> UnaryCall(const T& obj) {                     \
      if constexpr (This::HasUnaryMethod()) {                                \
        return MethodClassImpl<ValueT, T>{}.symbol_name(obj);                \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
                                                                             \
    static adt::Result<ValueT> HighOrderUnaryCall(                           \
        InterpreterBase<ValueT>* interpreter, const T& obj) {                \
      if constexpr (This::HasHighOrderUnaryMethod()) {                       \
        return MethodClassImpl<ValueT, T>{}.symbol_name(interpreter, obj);   \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
                                                                             \
    template <typename ObjT>                                                 \
    using BinaryMethodRetT =                                                 \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<const ObjT&>(), std::declval<const ValueT&>()));    \
                                                                             \
    template <typename ObjT>                                                 \
    using HighOrderBinaryMethodRetT =                                        \
        decltype(std::declval<MethodClassImpl<ValueT, ObjT>&>().symbol_name( \
            std::declval<InterpreterBase<ValueT>*>(),                        \
            std::declval<const ObjT&>(),                                     \
            std::declval<const ValueT&>()));                                 \
                                                                             \
    static constexpr bool HasBinaryMethod() {                                \
      return builtin_symbol::symbol_name::num_operands == 2 &&               \
             is_detected_v<BinaryMethodRetT, T>;                             \
    }                                                                        \
                                                                             \
    static constexpr bool HasHighOrderBinaryMethod() {                       \
      return builtin_symbol::symbol_name::num_operands == 2 &&               \
             is_detected_v<HighOrderBinaryMethodRetT, T>;                    \
    }                                                                        \
                                                                             \
    static adt::Result<ValueT> BinaryCall(const T& obj, const ValueT& arg) { \
      if constexpr (This::HasBinaryMethod()) {                               \
        return MethodClassImpl<ValueT, T>{}.symbol_name(obj, arg);           \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
    static adt::Result<ValueT> HighOrderBinaryCall(                          \
        InterpreterBase<ValueT>* interpreter,                                \
        const T& obj,                                                        \
        const ValueT& arg) {                                                 \
      if constexpr (This::HasHighOrderBinaryMethod()) {                      \
        return MethodClassImpl<ValueT, T>{}.symbol_name(                     \
            interpreter, obj, arg);                                          \
      } else {                                                               \
        return adt::errors::RuntimeError{"`" #symbol_name                    \
                                         "` method not found."};             \
      }                                                                      \
    }                                                                        \
  };

AXPR_FOR_EACH_SYMBOL_OP(SPECIALIZE_BuiltinMethodHelperImpl)

#undef SPECIALIZE_BuiltinMethodHelperImpl

template <typename VariantT, typename T>
struct DirectAlternative {
  static adt::Result<T> TryGet(const VariantT& val) {
    if (val.template Has<T>()) {
      return val.template Get<T>();
    }
    return adt::errors::TypeError{"cast failed."};
  }
};

template <typename ValueT, typename T>
struct IndirectAlternative {
  static adt::Result<T> TryGet(const ValueT& val) {
    using TypeT = typename TypeTrait<ValueT>::TypeT;
    ADT_LET_CONST_REF(type, DirectAlternative<ValueT, TypeT>::TryGet(val));
    return DirectAlternative<TypeT, T>::TryGet(type);
  }
};

template <typename ValueT,
          typename T,
          typename BuiltinSymbol,
          template <typename, typename>
          class Alternative>
struct BuiltinMethodHelper {
  using This = BuiltinMethodHelper;
  using Impl = BuiltinMethodHelperImpl<ValueT, T, BuiltinSymbol>;

  static constexpr bool HasUnaryMethod() { return Impl::HasUnaryMethod(); }

  static constexpr bool HasHighOrderUnaryMethod() {
    return Impl::HasHighOrderUnaryMethod();
  }

  static constexpr BuiltinUnaryFunc<ValueT> GetBuiltinUnaryMethod() {
    return &This::MakeBuiltinUnaryFunc<&Impl::UnaryCall>;
  }

  static constexpr BuiltinUnaryFunc<ValueT> GetBuiltinHighOrderUnaryMethod() {
    return &This::MakeBuiltinHighOrderUnaryFunc<&Impl::HighOrderUnaryCall>;
  }

  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc() {
    static const MethodClassImpl<ValueT, T>
        detect_specialization_of_method_class_impl;
    (void)detect_specialization_of_method_class_impl;
    if constexpr (HasUnaryMethod()) {
      return GetBuiltinUnaryMethod();
    } else if constexpr (HasHighOrderUnaryMethod()) {
      return GetBuiltinHighOrderUnaryMethod();
    } else if constexpr (HasDefaultUnaryMethod()) {
      return MethodClassImpl<ValueT,
                             T>::template GetBuiltinUnaryFunc<BuiltinSymbol>();
    } else {
      return adt::Nothing{};
    }
  }

  template <typename ObjT>
  using UnaryMethodRetT =
      decltype(MethodClassImpl<ValueT, ObjT>::template GetBuiltinUnaryFunc<
               BuiltinSymbol>());

  static constexpr bool HasDefaultUnaryMethod() {
    return is_detected_v<UnaryMethodRetT, T>;
  }

  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc() {
    static const MethodClassImpl<ValueT, T>
        detect_specialization_of_method_class_impl;
    (void)detect_specialization_of_method_class_impl;
    if constexpr (Impl::HasBinaryMethod()) {
      return &This::MakeBuiltinBinaryFunc<&Impl::BinaryCall>;
    } else if constexpr (Impl::HasHighOrderBinaryMethod()) {
      return &This::MakeBuiltinHighOrderBinaryFunc<&Impl::HighOrderBinaryCall>;
    } else if constexpr (HasDefaultBinaryMethod()) {
      return MethodClassImpl<ValueT,
                             T>::template GetBuiltinBinaryFunc<BuiltinSymbol>();
    } else {
      return adt::Nothing{};
    }
  }

  template <typename ObjT>
  using BinaryMethodRetT =
      decltype(MethodClassImpl<ValueT, ObjT>::template GetBuiltinBinaryFunc<
               BuiltinSymbol>());

  static constexpr bool HasDefaultBinaryMethod() {
    return is_detected_v<BinaryMethodRetT, T>;
  }

  template <adt::Result<ValueT> (*UnaryFunc)(const T&)>
  static adt::Result<ValueT> MakeBuiltinUnaryFunc(const ValueT& obj_val) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    const auto& ret = UnaryFunc(obj);
    return ret;
  }

  template <adt::Result<ValueT> (*UnaryFunc)(InterpreterBase<ValueT>*,
                                             const T&)>
  static adt::Result<ValueT> MakeBuiltinHighOrderUnaryFunc(
      InterpreterBase<ValueT>* interpreter, const ValueT& obj_val) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    const auto& ret = UnaryFunc(interpreter, obj);
    return ret;
  }

  template <adt::Result<ValueT> (*BinaryFunc)(const T&, const ValueT&)>
  static adt::Result<ValueT> MakeBuiltinBinaryFunc(const ValueT& obj_val,
                                                   const ValueT& arg) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    return BinaryFunc(obj, arg);
  }
  template <adt::Result<ValueT> (*BinaryFunc)(
      InterpreterBase<ValueT>*, const T&, const ValueT&)>
  static adt::Result<ValueT> MakeBuiltinHighOrderBinaryFunc(
      InterpreterBase<ValueT>* interpreter,
      const ValueT& obj_val,
      const ValueT& arg) {
    ADT_LET_CONST_REF(obj, Alternative<ValueT, T>::TryGet(obj_val));
    return BinaryFunc(interpreter, obj, arg);
  }
};

}  // namespace detail

template <typename ValueT>
struct MethodClass {
  using This = MethodClass;

  static BuiltinUnaryFunc<ValueT> Hash(const ValueT& val) {
    using S = builtin_symbol::Hash;
    return val.Match([](const auto& impl) -> BuiltinUnaryFunc<ValueT> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> BuiltinUnaryFunc<ValueT> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          if constexpr (Helper::HasUnaryMethod()) {
            return Helper::GetBuiltinUnaryMethod();
          } else if constexpr (Helper::HasHighOrderUnaryMethod()) {
            return Helper::GetBuiltinHighOrderUnaryMethod();
          } else {
            return &This::TypeDefaultHash<TT>;
          }
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        if constexpr (Helper::HasUnaryMethod()) {
          return Helper::GetBuiltinUnaryMethod();
        } else if constexpr (Helper::HasHighOrderUnaryMethod()) {
          return Helper::GetBuiltinHighOrderUnaryMethod();
        } else {
          return &This::InstanceDefaultHash<T>;
        }
      }
    });
  }

  template <typename TT>
  static adt::Result<ValueT> TypeDefaultHash(const ValueT& val) {
    int64_t hash_value = std::hash<const char*>()(typeid(TT).name());
    return hash_value;
  }

  template <typename T>
  static adt::Result<ValueT> InstanceDefaultHash(const ValueT& val) {
    ADT_LET_CONST_REF(impl, val.template TryGet<T>());
    // please implement MethodClassImpl<ValueT, T>::Hash if T is not defined
    // by ADT_DEFINE_RC.
    const void* ptr = impl.__adt_rc_shared_ptr_raw_ptr();
    return reinterpret_cast<int64_t>(ptr);
  }

  static BuiltinUnaryFunc<ValueT> ToString(const ValueT& val) {
    using S = builtin_symbol::ToString;
    return val.Match([](const auto& impl) -> BuiltinUnaryFunc<ValueT> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> BuiltinUnaryFunc<ValueT> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          if constexpr (Helper::HasUnaryMethod()) {
            return Helper::GetBuiltinUnaryMethod();
          } else if constexpr (Helper::HasHighOrderUnaryMethod()) {
            return Helper::GetBuiltinHighOrderUnaryMethod();
          } else {
            return &This::TypeDefaultToString<TT>;
          }
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        if constexpr (Helper::HasUnaryMethod()) {
          return Helper::GetBuiltinUnaryMethod();
        } else if constexpr (Helper::HasHighOrderUnaryMethod()) {
          return Helper::GetBuiltinHighOrderUnaryMethod();
        } else {
          return &This::InstanceDefaultToString<T>;
        }
      }
    });
  }

  template <typename TT>
  static adt::Result<ValueT> TypeDefaultToString(const ValueT& val) {
    std::ostringstream ss;
    ss << "<class '" << TT{}.Name() << "'>";
    return ss.str();
  }

  template <typename T>
  static adt::Result<ValueT> InstanceDefaultToString(const ValueT& val) {
    std::ostringstream ss;
    ADT_LET_CONST_REF(impl, val.template TryGet<T>());
    // please implement MethodClassImpl<ValueT, T>::ToString if T is not defined
    // by ADT_DEFINE_RC.
    const void* ptr = impl.__adt_rc_shared_ptr_raw_ptr();
    ss << "<" << TypeImpl<T>{}.Name() << " object at " << ptr << ">";
    return ss.str();
  }

  template <typename BuiltinUnarySymbol>
  static BuiltinUnaryFunc<ValueT> GetBuiltinUnaryFunc(const ValueT& val) {
    using S = BuiltinUnarySymbol;
    return val.Match([](const auto& impl) -> BuiltinUnaryFunc<ValueT> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> BuiltinUnaryFunc<ValueT> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          return Helper::GetBuiltinUnaryFunc();
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        return Helper::GetBuiltinUnaryFunc();
      }
    });
  }

  template <typename BuiltinBinarySymbol>
  static BuiltinBinaryFunc<ValueT> GetBuiltinBinaryFunc(const ValueT& val) {
    using S = BuiltinBinarySymbol;
    return val.Match([](const auto& impl) -> BuiltinBinaryFunc<ValueT> {
      using T = std::decay_t<decltype(impl)>;
      if constexpr (IsType<T>()) {
        return impl.Match([](const auto& type_impl)
                              -> BuiltinBinaryFunc<ValueT> {
          using TT = std::decay_t<decltype(type_impl)>;
          using Helper = detail::
              BuiltinMethodHelper<ValueT, TT, S, detail::IndirectAlternative>;
          return Helper::GetBuiltinBinaryFunc();
        });
      } else {
        using Helper = detail::
            BuiltinMethodHelper<ValueT, T, S, detail::DirectAlternative>;
        return Helper::GetBuiltinBinaryFunc();
      }
    });
  }
};

template <typename ValueT, typename T>
using __AltT = decltype(std::declval<ValueT&>().template Get<T>());

template <typename T, typename ValueT>
adt::Result<T> TryGetAlternative(const ValueT& val) {
  if constexpr (detail::is_detected_v<__AltT, ValueT, T>) {
    return val.template TryGet<T>();
  } else {
    return detail::IndirectAlternative<ValueT, T>::TryGet(val);
  }
}

template <typename T, typename ValueT>
adt::Result<T> TryGetImpl(const ValueT& val) {
  return TryGetAlternative<T, ValueT>(val);
}

template <typename ValueT>
std::string GetTypeName(const ValueT& val) {
  return val.Match(
      [](const BuiltinClassInstance<ValueT>& impl) -> std::string {
        return impl.type.class_attrs()->class_name;
      },
      [](const ClassInstance<ValueT>& impl) -> std::string {
        return impl->type.class_attrs->class_name;
      },
      [](const auto& impl) -> std::string {
        using T = std::decay_t<decltype(impl)>;
        return TypeImpl<T>{}.Name();
      });
}

}  // namespace ap::axpr

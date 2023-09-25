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

#include <memory>
#include <type_traits>

namespace pir {
///
/// \brief The template function actually called by isa_wrap.
///
template <typename Target, typename From, typename Enabler = void>
struct isa_impl {
  static inline bool call(const From &Val) { return Target::classof(Val); }
};

template <typename Target, typename From>
struct isa_impl<
    Target,
    From,
    typename std::enable_if<std::is_base_of<Target, From>::value>::type> {
  static inline bool call(const From &) { return true; }
};

///
/// \brief The template function actually called by isa.
///
template <typename Target, typename From, typename Enable = void>
struct isa_wrap {
  static inline bool call(const From &Val) {
    return isa_impl<Target, From>::call(Val);
  }
};

///
/// \brief typequalified specialization of the isa_wrap template parameter From.
/// Specialized types include: const T, T*, const T*, T* const, const T* const.
///
template <typename Target, typename From>
struct isa_wrap<Target, const From> {
  static inline bool call(const From &Val) {
    return isa_impl<Target, From>::call(Val);
  }
};

template <typename Target, typename From>
struct isa_wrap<
    Target,
    From,
    typename std::enable_if_t<std::is_pointer<std::decay_t<From>>::value>> {
  static inline bool call(
      std::remove_pointer_t<std::decay_t<From>> const *Val) {
    if (Val == nullptr) {
      throw("isa<> used on a null pointer");
    }
    return isa_impl<Target, std::remove_pointer_t<std::decay_t<From>>>::call(
        *Val);
  }
};

///
/// \brief isa template function, used to determine whether the value is a
/// Target type. Using method: if (isa<Target_Type>(value)) { ... }.
///
template <typename Target, typename From>
inline bool isa(const From &Val) {
  return isa_wrap<typename std::remove_pointer<Target>::type, From>::call(Val);
}

///
/// \brief Derive cast return type by template parameter From and To.
///
template <typename To, typename From>
struct ReturnTypeDuductionWrap {
  typedef To &type;
};

template <typename To, typename From>
struct ReturnTypeDuductionWrap<To, const From> {
  typedef const To &type;
};

template <typename To, typename From>
struct ReturnTypeDuductionWrap<To, From *> {
  typedef To *type;
};

template <typename To, typename From>
struct ReturnTypeDuductionWrap<To, const From *> {
  typedef const To *type;
};

template <typename To, typename From>
struct ReturnTypeDuductionWrap<To, const From *const> {
  typedef const To *type;
};

template <typename To, typename From>
struct ReturnTypeDuduction {
  typedef typename ReturnTypeDuductionWrap<To, From>::type type;
};

///
/// \brief cast From to To
///
template <typename To, typename From, typename Enable = void>
struct cast_impl {
  // This _is_ a simple type, just cast it.
  static typename ReturnTypeDuduction<To, From>::type call(const From &Val) {
    typename ReturnTypeDuduction<To, From>::type ret =
        (typename ReturnTypeDuduction<To, From>::type) const_cast<From &>(Val);
    return ret;
  }
};

template <typename To, typename From>
inline decltype(auto) cast(const From &Val) {
  if (!isa<To>(Val)) {
    throw("cast<To>() argument of incompatible type!");
  }
  return cast_impl<To, const From>::call(Val);
}

template <typename To, typename From>
inline decltype(auto) cast(From &Val) {  // NOLINT
  if (!isa<To>(Val)) {
    throw("cast<To>() argument of incompatible type!");
  }
  return cast_impl<To, From>::call(Val);
}

template <typename To, typename From>
inline decltype(auto) cast(From *Val) {
  if (!isa<To>(Val)) {
    throw("cast<To>() argument of incompatible type!");
  }
  return cast_impl<To, From *>::call(Val);
}

template <typename To, typename From>
inline decltype(auto) cast(std::unique_ptr<From> &&Val) {
  if (!isa<To>(Val)) {
    throw("cast<To>() argument of incompatible type!");
  }
  return cast_impl<To, std::unique_ptr<From>>::call(std::move(Val));
}

///
/// \brief dyn_cast From to To.
///
template <typename To, typename From>
inline decltype(auto) dyn_cast(const From &Val) {
  return isa<To>(Val) ? cast<To>(Val) : nullptr;
}

template <typename To, typename From>
inline decltype(auto) dyn_cast(From &Val) {  // NOLINT
  return isa<To>(Val) ? cast<To>(Val) : nullptr;
}

template <typename To, typename From>
inline decltype(auto) dyn_cast(From *Val) {
  return isa<To>(Val) ? cast<To>(Val) : nullptr;
}

template <typename To, typename From>
inline decltype(auto) dyn_cast(std::unique_ptr<From> &&Val) {
  return isa<To>(Val) ? cast<To>(std::forward<std::unique_ptr<From> &&>(Val))
                      : nullptr;
}

}  // namespace pir

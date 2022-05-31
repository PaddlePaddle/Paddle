/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <Python.h>

#include <vector>

#include "glog/logging.h"
#include "paddle/fluid/platform/variant.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
// Cast boost::variant for PyBind.
// Copy from
// https://github.com/pybind/pybind11/issues/576#issuecomment-269563199
namespace pybind11 {
namespace detail {

#if !defined(PYBIND11_HIDDEN)
#ifdef _WIN32
#define PYBIND11_HIDDEN __declspec(dllexport)
#else
#define PYBIND11_HIDDEN __attribute__((visibility("hidden")))
#endif
#endif

// Can be replaced by a generic lambda in C++14
struct PYBIND11_HIDDEN paddle_variant_caster_visitor
    : public boost::static_visitor<handle> {
  return_value_policy policy;
  handle parent;

  paddle_variant_caster_visitor(return_value_policy policy, handle parent)
      : policy(policy), parent(parent) {}

  template <class T,
            typename std::enable_if<!std::is_same<T, std::string>::value,
                                    bool>::type* = nullptr>
  handle operator()(T const& src) const {
    return make_caster<T>::cast(src, policy, parent);
  }

  template <class T,
            typename std::enable_if<std::is_same<T, std::string>::value,
                                    bool>::type* = nullptr>
  handle operator()(T const& src) const {
    try {
      return make_caster<T>::cast(src, policy, parent);
    } catch (std::exception& ex) {
      VLOG(4) << ex.what();
      VLOG(4) << src;
      // UnicodeDecodeError, src is not utf-8 encoded
      // see details:
      // https://github.com/pybind/pybind11/blob/master/docs/advanced/cast/strings.rst
      return PYBIND11_BYTES_FROM_STRING_AND_SIZE(src.data(), src.size());
    }
  }
};

template <class Variant>
struct paddle_variant_caster;

template <template <class...> class V, class... Ts>
struct paddle_variant_caster<V<Ts...>> {
  using Type = V<Ts...>;

  template <typename T>
  typename std::enable_if<
      !std::is_same<T, boost::detail::variant::void_>::value, bool>::type
  try_load(handle src, bool convert) {
    auto caster = make_caster<T>();
    if (!load_success_ && caster.load(src, convert)) {
      load_success_ = true;

      if (std::is_same<T, std::vector<float>>::value) {
        auto caster_ints = make_caster<std::vector<int64_t>>();
        if (caster_ints.load(src, convert)) {
          VLOG(4) << "This value are floats and int64_ts satisfy "
                     "simultaneously, will set it's type to "
                     "std::vector<int64_t>";
          value = cast_op<std::vector<int64_t>>(caster_ints);
          return true;
        }
      }

      if (std::is_same<T, float>::value) {
        auto caster_int64 = make_caster<int64_t>();
        if (caster_int64.load(src, convert)) {
          VLOG(4) << "this value are float and int64 satisfy simula.";
          value = cast_op<int64_t>(caster_int64);
          return true;
        }
      }

      value = cast_op<T>(caster);
      return true;
    }
    return false;
  }

  template <typename T>
  typename std::enable_if<std::is_same<T, boost::detail::variant::void_>::value,
                          bool>::type
  try_load(handle src, bool convert) {
    return false;
  }

  bool load(handle src, bool convert) {
    auto unused = {false, try_load<Ts>(src, convert)...};
    (void)(unused);
    return load_success_;
  }

  static handle cast(Type const& src, return_value_policy policy,
                     handle parent) {
    paddle_variant_caster_visitor visitor(policy, parent);
    return boost::apply_visitor(visitor, src);
  }

  PYBIND11_TYPE_CASTER(Type, _("Variant"));
  bool load_success_{false};
};

// Add specialization for concrete variant type
template <class... Args>
struct type_caster<boost::variant<Args...>>
    : paddle_variant_caster<boost::variant<Args...>> {};

}  // namespace detail
}  // namespace pybind11

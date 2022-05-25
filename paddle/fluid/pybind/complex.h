/*
    pybind11/complex.h: Complex number support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "paddle/phi/common/complex.h"
#include "pybind11/pybind11.h"

/// glibc defines I as a macro which breaks things, e.g., boost template names
#ifdef I
#undef I
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// template <typename T> struct format_descriptor<phi::dtype::complex<T>,
// detail::enable_if_t<std::is_floating_point<T>::value>> {
//     static constexpr const char c = format_descriptor<T>::c;
//     static constexpr const char value[3] = { 'Z', c, '\0' };
//     static std::string format() { return std::string(value); }
// };

// #ifndef PYBIND11_CPP17

// template <typename T> constexpr const char format_descriptor<
//     phi::dtype::complex<T>,
//     detail::enable_if_t<std::is_floating_point<T>::value>>::value[3];

// #endif

NAMESPACE_BEGIN(detail)

// template <typename T> struct is_fmt_numeric<std::complex<T>,
// detail::enable_if_t<std::is_floating_point<T>::value>> {
//     static constexpr bool value = true;
//     static constexpr int index = is_fmt_numeric<T>::index + 3;
// };

template <typename T>
class type_caster<phi::dtype::complex<T>> {
 public:
  bool load(handle src, bool convert) {
    if (!src) return false;
    if (!convert && !PyComplex_Check(src.ptr())) return false;
    Py_complex result = PyComplex_AsCComplex(src.ptr());
    if (result.real == -1.0 && PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    value = phi::dtype::complex<T>((T)result.real, (T)result.imag);
    return true;
  }

  static handle cast(const phi::dtype::complex<T> &src,
                     return_value_policy /* policy */, handle /* parent */) {
    return PyComplex_FromDoubles(static_cast<double>(src.real),
                                 static_cast<double>(src.imag));
  }

  PYBIND11_TYPE_CASTER(phi::dtype::complex<T>, _("complex"));
};
NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)

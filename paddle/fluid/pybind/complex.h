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

/*
    This file is adapted from
   https://github.com/pybind/pybind11/blob/master/include/pybind11/complex.h.
    The original license is kept as-is:

    pybind11/complex.h: Complex number support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <Python.h>
#include "paddle/phi/common/complex.h"
#include "pybind11/pybind11.h"

/// glibc defines I as a macro which breaks things, e.g., boost template names
#ifdef I
#undef I
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

// The specialization is added to make phi::dtype::complex<T> values
// casted as python complex values automatically when return from a function
// exported to python via pybind.
// For more details about custom type casters, see
// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
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
    value = phi::dtype::complex<T>(static_cast<T>(result.real),
                                   static_cast<T>(result.imag));
    return true;
  }

  static handle cast(const phi::dtype::complex<T> &src,
                     return_value_policy /* policy */,
                     handle /* parent */) {
    return PyComplex_FromDoubles(static_cast<double>(src.real),
                                 static_cast<double>(src.imag));
  }

  PYBIND11_TYPE_CASTER(phi::dtype::complex<T>, _("complex"));
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

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

#include <c10/core/DefaultDtype.h>
#include <c10/util/complex.h>
#include <c10/util/typeid.h>

namespace c10 {
static auto default_dtype = caffe2::TypeMeta::Make<float>();
static auto default_dtype_as_scalartype = default_dtype.toScalarType();
static auto default_complex_dtype =
    caffe2::TypeMeta::Make<c10::complex<float>>();

void set_default_dtype(caffe2::TypeMeta dtype) {
  default_dtype = dtype;
  default_dtype_as_scalartype = default_dtype.toScalarType();
  switch (default_dtype_as_scalartype) {
    case ScalarType::Half:
      default_complex_dtype = ScalarType::ComplexHalf;
      break;
    case ScalarType::Double:
      default_complex_dtype = ScalarType::ComplexDouble;
      break;
    default:
      default_complex_dtype = ScalarType::ComplexFloat;
      break;
  }
}

const caffe2::TypeMeta get_default_dtype() { return default_dtype; }

ScalarType get_default_dtype_as_scalartype() {
  return default_dtype_as_scalartype;
}

const caffe2::TypeMeta get_default_complex_dtype() {
  return default_complex_dtype;
}
}  // namespace c10

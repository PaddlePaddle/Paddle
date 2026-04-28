// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>

#include "paddle/common/macros.h"

namespace c10 {
PADDLE_API void set_default_dtype(caffe2::TypeMeta dtype);
PADDLE_API const caffe2::TypeMeta get_default_dtype();
PADDLE_API ScalarType get_default_dtype_as_scalartype();
inline const caffe2::TypeMeta get_default_complex_dtype() {
  switch (get_default_dtype_as_scalartype()) {
    case ScalarType::Half:
      return caffe2::TypeMeta::fromScalarType(ScalarType::ComplexHalf);
    case ScalarType::Double:
      return caffe2::TypeMeta::fromScalarType(ScalarType::ComplexDouble);
    default:
      return caffe2::TypeMeta::fromScalarType(ScalarType::ComplexFloat);
  }
}
}  // namespace c10

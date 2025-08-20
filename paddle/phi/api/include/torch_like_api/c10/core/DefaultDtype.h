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

namespace c10 {
static auto default_dtype = ScalarType::Float;
static auto default_complex_dtype = ScalarType::ComplexFloat;

void inline set_default_dtype(ScalarType dtype) { default_dtype = dtype; }

const ScalarType inline get_default_dtype() { return default_dtype; }

ScalarType inline get_default_dtype_as_scalartype() { return default_dtype; }

const ScalarType inline get_default_complex_dtype() {
  return default_complex_dtype;
}
}  // namespace c10

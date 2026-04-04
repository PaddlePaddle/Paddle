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

#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>
#include <optional>

namespace c10 {

/// Convert ScalarType enum values to TypeMeta handles.
inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
  return caffe2::TypeMeta::fromScalarType(scalar_type);
}

/// Convert TypeMeta handles to ScalarType enum values.
inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  return dtype.toScalarType();
}

/// typeMetaToScalarType(), lifted to optional.
inline std::optional<ScalarType> optTypeMetaToScalarType(
    std::optional<caffe2::TypeMeta> type_meta) {
  if (!type_meta.has_value()) {
    return std::nullopt;
  }
  return type_meta->toScalarType();
}

/// Equality across TypeMeta / ScalarType.
inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  return m.isScalarType(t);
}
inline bool operator==(caffe2::TypeMeta m, ScalarType t) { return t == m; }
inline bool operator!=(ScalarType t, caffe2::TypeMeta m) { return !(t == m); }
inline bool operator!=(caffe2::TypeMeta m, ScalarType t) { return !(t == m); }

}  // namespace c10

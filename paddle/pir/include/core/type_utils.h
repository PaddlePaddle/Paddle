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
///
/// \brief Utility Functions
///

#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/builtin_type_interfaces.h"
#include "paddle/pir/include/core/dll_decl.h"

namespace pir {
///
/// \brief Return the element type or return the type itself.
///
Type GetElementTypeOrSelf(Type type);

///
/// \brief Returns true if the given two shapes are compatible. That is, they
/// have the same size and each pair of the elements are equal or one of them is
/// dynamic.
///
bool VerifyCompatibleShape(const pir::DDim& lhs_shape,
                           const pir::DDim& rhs_shape);

///
/// \brief Returns true if the given two types have compatible shape. That
/// is, they are both scalars (not shaped), or they are both shaped types and at
/// least one is unranked or they have compatible dimensions. Dimensions are
/// compatible if at least one is dynamic or both are equal. The element type
/// does not matter.
///
bool VerifyCompatibleShape(Type lhs_type, Type rhs_type);

///
/// \brief Dimensions are compatible if all non-dynamic dims are equal.
///
bool VerifyCompatibleDims(const std::vector<int64_t>& dims);

///
/// \brief Returns true if the given two arrays have the same number of elements
/// and each pair wise entries have compatible shape.
///
bool IR_API VerifyCompatibleShapes(const std::vector<Type>& lhs_types,
                                   const std::vector<Type>& rhs_types);

///
/// \brief Returns true if all given types have compatible shapes. That is,
/// they are all scalars (not shaped), or they are all shaped types and any
/// ranked shapes have compatible dimensions. Dimensions are compatible if all
/// non-dynamic dims are equal. The element type does not matter.
///
bool IR_API VerifyCompatibleShapes(const std::vector<Type>& types);
}  // namespace pir

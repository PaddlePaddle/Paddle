// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
/**
 * \file This file implements some intrinsic types used in CodeGen.
 */

#include "paddle/cinn/common/common.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {

/**
 * Type representation for cinn_buffer_t.
 */
struct BufferType {
  static BufferType Create(const Type& primitive) {
    return BufferType(primitive);
  }

  static Type cinn_type();

 private:
  explicit BufferType(const Type& primitive_type)
      : primitive_type(primitive_type) {
    PADDLE_ENFORCE_EQ(primitive_type.valid() && primitive_type.is_primitive(),
                      true,
                      ::common::errors::InvalidArgument(
                          "primitive type should be valid and primitive."));
  }

  //! Determine the primitive of cinn_buffer_t.
  Type primitive_type;
  static char c_type_repr[];
};

static Type make_intrinsic_buffer_type(Type primitive_type) {
  PADDLE_ENFORCE_EQ(primitive_type.valid() && primitive_type.is_primitive(),
                    true,
                    ::common::errors::InvalidArgument(
                        "primitive type should be valid and primitive."));
  Type res = BufferType::cinn_type();
  return res;
}

}  // namespace runtime
}  // namespace cinn

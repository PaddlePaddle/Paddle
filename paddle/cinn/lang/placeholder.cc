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

#include "paddle/cinn/lang/placeholder.h"

#include "paddle/cinn/ir/ir_utils.h"
#include "paddle/cinn/runtime/intrinsic.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace lang {

using cinn::common::bfloat16;
using cinn::common::float16;

ir::Tensor CreatePlaceHolder(const std::vector<int> &shape,
                             Type type,
                             const std::string &name) {
  std::vector<Expr> expr_shape;
  for (int s : shape) {
    expr_shape.push_back(Expr(s));
  }
  return CreatePlaceHolder(
      ir::utils::GetCompatibleShape(expr_shape), type, name);
}

ir::Tensor CreatePlaceHolder(const std::vector<ir::Dim> &shape,
                             Type type,
                             const std::string &name) {
  PADDLE_ENFORCE_GT(shape.size(),
                    0,
                    ::common::errors::PreconditionNotMet(
                        "The shape of Placeholder should not be empty."));
  if (type.is_float(32)) {
    return Placeholder<float>(name, shape);
  } else if (type.is_float(64)) {
    return Placeholder<double>(name, shape);
  } else if (type.is_bfloat16()) {
    return Placeholder<bfloat16>(name, shape);
  } else if (type.is_float16()) {
    return Placeholder<float16>(name, shape);
  } else if (type.is_int(8)) {
    return Placeholder<int8_t>(name, shape);
  } else if (type.is_int(16)) {
    return Placeholder<int16_t>(name, shape);
  } else if (type.is_int(32)) {
    return Placeholder<int32_t>(name, shape);
  } else if (type.is_int(64)) {
    return Placeholder<int64_t>(name, shape);
  } else if (type.is_uint(8)) {
    return Placeholder<uint8_t>(name, shape);
  } else if (type.is_uint(16)) {
    return Placeholder<uint16_t>(name, shape);
  } else if (type.is_uint(32)) {
    return Placeholder<uint32_t>(name, shape);
  } else if (type.is_uint(64)) {
    return Placeholder<uint64_t>(name, shape);
  } else if (type.is_bool()) {
    return Placeholder<bool>(name, shape);
  }
  CINN_NOT_IMPLEMENTED
}

ir::Tensor CreatePlaceHolder(const std::vector<Expr> &origin_shape,
                             Type type,
                             const std::string &name) {
  const auto shape = ir::utils::GetCompatibleShape(origin_shape);
  if (type.is_float(32)) {
    return Placeholder<float>(name, shape);
  } else if (type.is_float(64)) {
    return Placeholder<double>(name, shape);
  } else if (type.is_bfloat16()) {
    return Placeholder<bfloat16>(name, shape);
  } else if (type.is_float16()) {
    return Placeholder<float16>(name, shape);
  } else if (type.is_int(8)) {
    return Placeholder<int8_t>(name, shape);
  } else if (type.is_int(16)) {
    return Placeholder<int16_t>(name, shape);
  } else if (type.is_int(32)) {
    return Placeholder<int32_t>(name, shape);
  } else if (type.is_int(64)) {
    return Placeholder<int64_t>(name, shape);
  } else if (type.is_uint(8)) {
    return Placeholder<uint8_t>(name, shape);
  } else if (type.is_uint(16)) {
    return Placeholder<uint16_t>(name, shape);
  } else if (type.is_uint(32)) {
    return Placeholder<uint32_t>(name, shape);
  } else if (type.is_uint(64)) {
    return Placeholder<uint64_t>(name, shape);
  } else if (type.is_bool()) {
    return Placeholder<bool>(name, shape);
  }
  CINN_NOT_IMPLEMENTED
}

}  // namespace lang
}  // namespace cinn

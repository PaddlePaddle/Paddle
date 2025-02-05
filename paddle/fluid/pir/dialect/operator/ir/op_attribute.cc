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

#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"

namespace paddle::dialect {
const phi::IntArray &IntArrayAttribute::data() const {
  return storage()->GetAsKey();
}

phi::DataType DataTypeAttribute::data() const { return storage()->GetAsKey(); }

phi::Place PlaceAttribute::data() const { return storage()->GetAsKey(); }

phi::DataLayout DataLayoutAttribute::data() const {
  return storage()->GetAsKey();
}

phi::Scalar ScalarAttribute::data() const {
  if (isa<pir::FloatAttribute>()) {
    return phi::Scalar(dyn_cast<pir::FloatAttribute>().data());
  } else if (isa<pir::DoubleAttribute>()) {
    return phi::Scalar(dyn_cast<pir::DoubleAttribute>().data());
  } else if (isa<pir::Int32Attribute>()) {
    return phi::Scalar(dyn_cast<pir::Int32Attribute>().data());
  } else if (isa<pir::IndexAttribute>()) {
    return phi::Scalar(dyn_cast<pir::IndexAttribute>().data());
  } else if (isa<pir::Int64Attribute>()) {
    return phi::Scalar(dyn_cast<pir::Int64Attribute>().data());
  } else if (isa<pir::BoolAttribute>()) {
    return phi::Scalar(dyn_cast<pir::BoolAttribute>().data());
  } else if (isa<pir::StrAttribute>()) {
    return phi::Scalar(dyn_cast<pir::StrAttribute>().AsString());
  } else if (isa<pir::Complex64Attribute>()) {
    return phi::Scalar(dyn_cast<pir::Complex64Attribute>().data());
  } else if (isa<pir::Complex128Attribute>()) {
    return phi::Scalar(dyn_cast<pir::Complex128Attribute>().data());
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported ir attribute when casting it into "
        "phi scalar."));
  }
}
}  // namespace paddle::dialect
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)

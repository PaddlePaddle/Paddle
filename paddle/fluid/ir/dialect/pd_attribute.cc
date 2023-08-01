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

#include "paddle/fluid/ir/dialect/pd_attribute.h"

namespace paddle {
namespace dialect {
const phi::IntArray& IntArrayAttribute::data() const {
  return storage()->GetAsKey();
}

phi::DataType DataTypeAttribute::data() const { return storage()->GetAsKey(); }

phi::Place PlaceAttribute::data() const { return storage()->GetAsKey(); }

phi::DataLayout DataLayoutAttribute::data() const {
  return storage()->GetAsKey();
}

phi::Scalar ScalarAttribute::data() {
  if (isa<ir::FloatAttribute>()) {
    return phi::Scalar(dyn_cast<ir::FloatAttribute>().data());
  } else if (isa<ir::DoubleAttribute>()) {
    return phi::Scalar(dyn_cast<ir::DoubleAttribute>().data());
  } else if (isa<ir::Int32Attribute>()) {
    return phi::Scalar(dyn_cast<ir::Int32Attribute>().data());
  } else if (isa<ir::Int64Attribute>()) {
    return phi::Scalar(dyn_cast<ir::Int64Attribute>().data());
  } else if (isa<ir::BoolAttribute>()) {
    return phi::Scalar(dyn_cast<ir::BoolAttribute>().data());
  } else if (isa<ir::StrAttribute>()) {
    return phi::Scalar(dyn_cast<ir::StrAttribute>().AsString());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir attribute when casting it into "
        "phi scalar."));
  }
}

}  // namespace dialect
}  // namespace paddle

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)

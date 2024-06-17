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
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir attribute when casting it into "
        "phi scalar."));
  }
}

IntArrayAttribute IntArrayAttribute::Parse(pir::IrParser &parser) {  // NOLINT
  Token buket_token = parser.ConsumeToken();
  std::vector<int> vec{};
  while (parser.PeekToken().val_ != "]") {
    Token val_token = parser.ConsumeToken();
    vec.push_back(atoi(val_token.val_.c_str()));
    if (parser.PeekToken().val_ == "]") break;
    parser.ConsumeToken();
  }
  parser.ConsumeToken();
  return IntArrayAttribute::get(parser.ctx, vec);
}

// Parse a DataTypeAttribute
// DataTypeAttribute :=  bool|uint8|int8|uint16|int16|uint32
//                       |int32|uint64|int64|float32|complex64
//                       |complex128|Undefined|psting|flaot16
//                       |bfloat16|num_data_types|all_dtype
DataTypeAttribute DataTypeAttribute::Parse(pir::IrParser &parser) {  // NOLINT
  std::string datatype_token_val = parser.ConsumeToken().val_;
  PADDLE_ENFORCE_EQ(StringToDataTypeMap().count(datatype_token_val) > 0,
                    true,
                    common::errors::InvalidArgument(
                        datatype_token_val + " is not defined in DataType." +
                        parser.GetErrorLocationInfo()));
  return DataTypeAttribute::get(parser.ctx,
                                StringToDataTypeMap().at(datatype_token_val));
}

// Parse a PlaceAttribute
// PlaceAttribute   :=    Place(cpu)|Place(gpu:0)|Place(gpu_pinned)
//                        |Place(xpu:0)|Place(ipu:0)|Place(:0)|undefined
PlaceAttribute PlaceAttribute::Parse(pir::IrParser &parser) {  // NOLINT
  parser.ConsumeAToken("Place");
  parser.ConsumeAToken("(");
  std::string place_token_val = parser.ConsumeToken().val_;
  PADDLE_ENFORCE_EQ(StringToPlaceMap().count(place_token_val) > 0,
                    true,
                    common::errors::InvalidArgument(
                        place_token_val + " is not defined in Place." +
                        parser.GetErrorLocationInfo()));
  if (parser.PeekToken().val_ == ":") {
    parser.ConsumeAToken(":");
    parser.ConsumeToken();
  } else if (place_token_val == ":") {
    parser.ConsumeToken();
  }
  parser.ConsumeAToken(")");
  return PlaceAttribute::get(parser.ctx,
                             StringToPlaceMap().at(place_token_val));
}

// Parse a DataLayoutAttribute
// DataLayoutAttribute  :=   NHWC|NCHW|Undefined(0)|ONEDNN
//                           |SPARSE_COO|SPARSE_CSR|NDHWC
//                           |NCDHW|PSTRING_UNION|STRIDED
DataLayoutAttribute DataLayoutAttribute::Parse(
    pir::IrParser &parser) {  // NOLINT
  std::string datalayout_token_val = parser.ConsumeToken().val_;
  PADDLE_ENFORCE_EQ(
      StringToDataLayoutMap().count(datalayout_token_val) > 0,
      true,
      common::errors::InvalidArgument(datalayout_token_val +
                                      " is not defined in DataLayout." +
                                      parser.GetErrorLocationInfo()));
  if (datalayout_token_val == "Undefined") {
    parser.ConsumeAToken("(");
    parser.ConsumeAToken("AnyLayout");
    parser.ConsumeAToken(")");
  }
  return DataLayoutAttribute::get(
      parser.ctx, StringToDataLayoutMap().at(datalayout_token_val));
}

}  // namespace paddle::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)

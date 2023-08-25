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

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_attribute_storage.h"
#include "paddle/ir/core/attribute.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_parser.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"

namespace paddle {
namespace dialect {
class IntArrayAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(IntArrayAttribute,
                                    IntArrayAttributeStorage);

  bool operator<(const IntArrayAttribute &right) const {
    return storage() < right.storage();
  }

  static IntArrayAttribute Parse(ir::IrParser &parser) {  // NOLINT
    Token buket_token = parser.GetToken();
    std::vector<int32_t> vec{};
    while (parser.PeekToken().val_ != "]") {
      Token val_token = parser.GetToken();
      vec.push_back(atoll(val_token.val_.c_str()));
      if (parser.PeekToken().val_ == "]") break;
      parser.GetToken();
    }
    parser.GetToken();
    return IntArrayAttribute::get(parser.ctx, vec);
  }

  const phi::IntArray &data() const;
};

class ScalarAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  static bool classof(ir::Attribute val) {
    return (val.type_id() == ir::BoolAttribute::type_id()) ||
           (val.type_id() == ir::FloatAttribute::type_id()) ||
           (val.type_id() == ir::DoubleAttribute::type_id()) ||
           (val.type_id() == ir::Int32Attribute::type_id()) ||
           (val.type_id() == ir::Int64Attribute::type_id());
  }

  phi::Scalar data();
};

class DataTypeAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataTypeAttribute,
                                    DataTypeAttributeStorage);

  bool operator<(const DataTypeAttribute &right) const {
    return storage() < right.storage();
  }
  static DataTypeAttribute Parse(ir::IrParser &parser) {  // NOLINT
    Token datatype_token = parser.GetToken();
    if (datatype_token.val_ == "bool") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::BOOL);
    } else if (datatype_token.val_ == "uint8") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UINT8);
    } else if (datatype_token.val_ == "int8") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::INT8);
    }
    if (datatype_token.val_ == "uint16") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UINT16);
    } else if (datatype_token.val_ == "int16") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::INT16);
    } else if (datatype_token.val_ == "uint32") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UINT32);
    }
    if (datatype_token.val_ == "int32") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::INT32);
    } else if (datatype_token.val_ == "uint64") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UINT64);
    } else if (datatype_token.val_ == "int64") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::INT64);
    } else if (datatype_token.val_ == "float32") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::FLOAT32);
    } else if (datatype_token.val_ == "float64") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::FLOAT64);
    } else if (datatype_token.val_ == "complex64") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::COMPLEX64);
    } else if (datatype_token.val_ == "complex128") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::COMPLEX128);
    } else if (datatype_token.val_ == "undefined") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UNDEFINED);
    } else if (datatype_token.val_ == "psting") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::PSTRING);
    } else if (datatype_token.val_ == "float16") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::FLOAT16);
    } else if (datatype_token.val_ == "bfloat16") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::BFLOAT16);
    } else if (datatype_token.val_ == "num_data_types") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::NUM_DATA_TYPES);
    } else if (datatype_token.val_ == "all_dtype") {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::ALL_DTYPE);
    } else {
      return DataTypeAttribute::get(parser.ctx, phi::DataType::UNDEFINED);
    }
  }
  phi::DataType data() const;
};

class PlaceAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(PlaceAttribute, PlaceAttributeStorage);

  bool operator<(const PlaceAttribute &right) const {
    return storage() < right.storage();
  }
  static PlaceAttribute Parse(ir::IrParser &parser) {  // NOLINT
    Token place_identifier_token = parser.GetToken();
    Token left_parenthesis_token = parser.GetToken();
    Token place_token = parser.GetToken();
    if (place_token.val_ == "cpu") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::CPUPlace{});
    } else if (place_token.val_ == "gpu") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::GPUPlace{});
    } else if (place_token.val_ == "gpupinned") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::GPUPinnedPlace{});
    } else if (place_token.val_ == "xpu") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::XPUPlace{});
    } else if (place_token.val_ == "ipu") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::IPUPlace{});
    } else if (place_token.val_ == "custom") {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::CustomPlace{});
    } else if (place_token.val_ == "undefined") {
      parser.GetToken();
      parser.GetToken();
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::Place{});
    } else {
      parser.GetToken();
      return PlaceAttribute::get(parser.ctx, phi::Place{});
    }
  }
  phi::Place data() const;
};

class DataLayoutAttribute : public ir::Attribute {
 public:
  using Attribute::Attribute;

  DECLARE_ATTRIBUTE_UTILITY_FUNCTOR(DataLayoutAttribute,
                                    DataLayoutAttributeStorage);

  bool operator<(const DataLayoutAttribute &right) const {
    return storage() < right.storage();
  }
  static DataLayoutAttribute Parse(ir::IrParser &parser) {  // NOLINT
    Token data_layout_token = parser.GetToken();
    if (data_layout_token.val_ == "UNDEFINED") {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::UNDEFINED);
    } else if ((data_layout_token.val_ == "ANY") ||
               (data_layout_token.val_ == "kAnyLayout")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::ANY);
    } else if ((data_layout_token.val_ == "NCHW") ||
               (data_layout_token.val_ == "kNCHW")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::NCHW);
    } else if ((data_layout_token.val_ == "NHWC") ||
               (data_layout_token.val_ == "kNHWC")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::NHWC);
    } else if ((data_layout_token.val_ == "NCDHW") ||
               (data_layout_token.val_ == "kNCDHW")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::NCDHW);
    } else if ((data_layout_token.val_ == "NDHWC") ||
               (data_layout_token.val_ == "kNDHWC")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::NDHWC);
    } else if ((data_layout_token.val_ == "ONEDNN") ||
               (data_layout_token.val_ == "kMKLDNN")) {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::ONEDNN);
    } else if (data_layout_token.val_ == "SPARSE_COO") {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::SPARSE_COO);
    } else if (data_layout_token.val_ == "SPARSE_CSR") {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::SPARSE_CSR);
    } else if (data_layout_token.val_ == "PSTRING_UNION") {
      return DataLayoutAttribute::get(parser.ctx,
                                      phi::DataLayout::PSTRING_UNION);
    } else if (data_layout_token.val_ == "NUM_DATA_LAYOUTS") {
      return DataLayoutAttribute::get(parser.ctx,
                                      phi::DataLayout::NUM_DATA_LAYOUTS);
    } else if (data_layout_token.val_ == "ALL_LAYOUT") {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::ALL_LAYOUT);
    } else {
      return DataLayoutAttribute::get(parser.ctx, phi::DataLayout::UNDEFINED);
    }
  }

  phi::DataLayout data() const;
};

}  // namespace dialect
}  // namespace paddle

IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::IntArrayAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::ScalarAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataTypeAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::PlaceAttribute)
IR_DECLARE_EXPLICIT_TYPE_ID(paddle::dialect::DataLayoutAttribute)

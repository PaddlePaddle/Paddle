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

#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

namespace paddle {
namespace dialect {

const std::unordered_set<std::string> LegacyOpList = {
    "pd_op.load_combine",
    "pd_op.c_concat",
    "pd_op.c_broadcast_",
    "pd_op.fused_bn_add_activation_",
    "pd_op.fused_bn_add_activation_grad",
    "pd_op.c_sync_calc_stream_",
    "pd_op.c_sync_comm_stream_",
    "pd_op.send_v2",
    "pd_op.recv_v2",
    "pd_op.c_allreduce_sum",
    "pd_op.c_allreduce_sum_",
    "pd_op.c_reduce_sum",
    "pd_op.c_reduce_sum_",
    "pd_op.c_allreduce_max_",
    "pd_op.c_allgather",
    "pd_op.seed",
    "pd_op.share_data"};

enum class AttrType {
  UNDEFINED = 0,
  BOOL,
  INT32,
  INT64,

  FLOAT,
  DOUBLE,

  ARRAY,
  INT_ARRAY,

  SCALAR,
  DATA_TYPE,
  DATA_LAYOUT,
  PLACE,

  STRING,

  NUM_ATTR_TYPES,
};

static inline AttrType GetAttributeType(const pir::Attribute& attr) {
  if (attr.isa<pir::BoolAttribute>()) {
    return AttrType::BOOL;
  } else if (attr.isa<pir::FloatAttribute>()) {
    return AttrType::FLOAT;
  } else if (attr.isa<pir::DoubleAttribute>()) {
    return AttrType::DOUBLE;
  } else if (attr.isa<pir::Int32Attribute>()) {
    return AttrType::INT32;
  } else if (attr.isa<pir::Int64Attribute>()) {
    return AttrType::INT64;
  } else if (attr.isa<pir::ArrayAttribute>()) {
    return AttrType::ARRAY;
  } else if (attr.isa<pir::StrAttribute>()) {
    return AttrType::STRING;
  } else if (attr.isa<paddle::dialect::IntArrayAttribute>()) {
    return AttrType::INT_ARRAY;
  } else if (attr.isa<paddle::dialect::DataTypeAttribute>()) {
    return AttrType::DATA_TYPE;
  } else if (attr.isa<paddle::dialect::PlaceAttribute>()) {
    return AttrType::PLACE;
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported ir Attribute type when casting it into "
        "AttrType."));
  }
}

static std::unordered_map<
    AttrType,
    std::function<VariantType(const pir::Attribute& attr)>>
    kAttrCastMap = {
        {AttrType::BOOL,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::BoolAttribute>().data()};
         }},
        {AttrType::FLOAT,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::FloatAttribute>().data()};
         }},
        {AttrType::DOUBLE,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::DoubleAttribute>().data()};
         }},
        {AttrType::INT32,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::Int32Attribute>().data()};
         }},
        {AttrType::INT64,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::Int64Attribute>().data()};
         }},
        {AttrType::INT_ARRAY,
         [](const pir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::IntArrayAttribute>()
                   .data()
                   .GetData()};
         }},
        {AttrType::STRING,
         [](const pir::Attribute& attr) {
           return VariantType{attr.dyn_cast<pir::StrAttribute>().AsString()};
         }},
        {AttrType::DATA_TYPE,
         [](const pir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::DataTypeAttribute>().data()};
         }},
        {AttrType::PLACE,
         [](const pir::Attribute& attr) {
           return VariantType{
               attr.dyn_cast<paddle::dialect::PlaceAttribute>().data()};
         }},
        {AttrType::ARRAY,
         [](const pir::Attribute& attr) {
           auto attr_vec = attr.dyn_cast<pir::ArrayAttribute>().AsVector();
           if (attr_vec.size() == 0) {
             return VariantType{std::vector<int>()};
           }
           AttrType element_type = GetAttributeType(attr_vec[0]);

           if (element_type == AttrType::BOOL) {
             std::vector<bool> vec_bools;
             for (auto vec_element : attr_vec) {
               vec_bools.push_back(
                   vec_element.dyn_cast<pir::BoolAttribute>().data());
             }
             return VariantType{vec_bools};
           } else if (element_type == AttrType::INT32) {
             std::vector<int> vec_int32;
             for (auto vec_element : attr_vec) {
               vec_int32.push_back(
                   vec_element.dyn_cast<pir::Int32Attribute>().data());
             }
             return VariantType{vec_int32};
           } else if (element_type == AttrType::INT64) {
             std::vector<int64_t> vec_int64;
             for (auto vec_element : attr_vec) {
               vec_int64.push_back(
                   vec_element.dyn_cast<pir::Int64Attribute>().data());
             }
             return VariantType{vec_int64};
           } else if (element_type == AttrType::FLOAT) {
             std::vector<float> vec_float;
             for (auto vec_element : attr_vec) {
               vec_float.push_back(
                   vec_element.dyn_cast<pir::FloatAttribute>().data());
             }
             return VariantType{vec_float};
           } else if (element_type == AttrType::DOUBLE) {
             std::vector<double> vec_double;
             for (auto vec_element : attr_vec) {
               vec_double.push_back(
                   vec_element.dyn_cast<pir::DoubleAttribute>().data());
             }
             return VariantType{vec_double};
           } else if (element_type == AttrType::STRING) {
             std::vector<std::string> vec_string;
             for (auto vec_element : attr_vec) {
               vec_string.push_back(
                   vec_element.dyn_cast<pir::StrAttribute>().AsString());
             }
             return VariantType{vec_string};
           } else {
             PADDLE_THROW(phi::errors::Unimplemented(
                 "Unsupported ir Attribute type when casting it into "
                 "vector."));
           }
         }},
};

VariantType GetAttributeData(const pir::Attribute& attr) {
  AttrType attr_type = GetAttributeType(attr);
  return kAttrCastMap[attr_type](attr);
}

bool IsLegacyOp(const std::string& name) { return LegacyOpList.count(name); }

bool IsEmptyOpResult(const pir::OpResult& op_result) {
  return !op_result.impl() || op_result.type().isa<pir::Type>();
}

}  // namespace dialect
}  // namespace paddle

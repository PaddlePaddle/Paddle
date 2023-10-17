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

#include <glog/logging.h>
#include <unordered_set>

#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/builtin_type.h"

namespace paddle {
namespace dialect {

const std::unordered_set<std::string> LegacyOpList = {
    "pd_op.load_combine",
    "pd_op.c_concat",
    "pd_op.c_broadcast_",
    "pd_op.c_sync_calc_stream_",
    "pd_op.c_sync_comm_stream_",
    "pd_op.dpsgd",
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

bool IsEmptyValue(const pir::Value& value) {
  return !value.impl() || !value.type();
}

std::set<std::string> GetRegisterDataType(const std::string& op_name) {
  std::set<framework::proto::VarType::Type> proto_type;

  auto phi_kernels = phi::KernelFactory::Instance().kernels();
  for (auto& kernel_pair : phi_kernels) {
    auto fluid_op_name = phi::TransToFluidOpName(kernel_pair.first);
    if (kernel_pair.first != op_name && fluid_op_name != op_name) {
      continue;
    }
    for (auto& info_pair : kernel_pair.second) {
      framework::OpKernelType kernel_type =
          framework::TransPhiKernelKeyToOpKernelType(info_pair.first);
      proto_type.insert(kernel_type.data_type_);
    }
  }

  std::set<std::string> data_type;
  for (auto& iter : proto_type) {
    data_type.insert(
        phi::DataTypeToString(framework::TransToPhiDataType(iter)));
  }

  VLOG(1) << "data_type.size(): " << data_type.size();

  for (auto& type : data_type) {
    VLOG(1) << "data_type: " << type;
  }

  return data_type;
}

void CheckDtype(const pir::Value& value,
                const std::string& input_name,
                const std::string& op_name) {
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);

  if (value.type().isa<pir::DenseTensorType>()) {
    std::string value_type = phi::DataTypeToString(dialect::TransToPhiDataType(
        value.type().dyn_cast<pir::DenseTensorType>().dtype()));
    if (expected_dtype.find(value_type) == expected_dtype.end()) {
      PADDLE_THROW(phi::errors::InvalidArgument("Type error %s.", input_name));
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get dtype for dense "
        "tensor."));
  }
}

}  // namespace dialect
}  // namespace paddle

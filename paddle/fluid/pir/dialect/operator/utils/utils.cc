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
#include <sstream>
#include <unordered_set>

#include "paddle/fluid/framework/phi_utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/utils/string/string_helper.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_api.h"

namespace paddle {
namespace dialect {

const std::unordered_set<std::string> LegacyOpList = {
    LoadCombineOp::name(),
    CConcatOp::name(),
    CBroadcast_Op::name(),
    CSyncCalcStream_Op::name(),
    CSyncCommStream_Op::name(),
    FusedElemwiseAddActivationOp::name(),
    FusedElemwiseAddActivationGradOp::name(),
    FusedGemmEpilogueOp::name(),
    FusedGemmEpilogueGradOp::name(),
    DpsgdOp::name(),
    SendV2Op::name(),
    RecvV2Op::name(),
    CAllreduceSumOp::name(),
    CAllreduceSum_Op::name(),
    CReduceSumOp::name(),
    CReduceSum_Op::name(),
    CAllreduceMax_Op::name(),
    CAllgatherOp::name(),
    SeedOp::name(),
    ShareDataOp::name(),
    SparseMomentumOp::name()};

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

std::vector<int64_t> GetInt64Vector(const pir::Attribute& attr) {
  PADDLE_ENFORCE_EQ(attr.isa<pir::ArrayAttribute>(),
                    true,
                    phi::errors::PreconditionNotMet(
                        "attribute MUST be a pir::ArrayAttribute"));
  auto attr_vec = attr.dyn_cast<pir::ArrayAttribute>().AsVector();

  std::vector<int64_t> vec_int64;
  for (auto vec_element : attr_vec) {
    PADDLE_ENFORCE_EQ(
        vec_element.isa<pir::Int64Attribute>(),
        true,
        phi::errors::PreconditionNotMet("element MUST be a Int64Attribute"));
    vec_int64.push_back(vec_element.dyn_cast<pir::Int64Attribute>().data());
  }

  return vec_int64;
}

std::set<std::string> GetRegisterDataType(const std::string& op_name) {
  std::string non_inplace_op_name;
  if (paddle::string::ends_with(op_name, "_")) {
    non_inplace_op_name = op_name.substr(0, op_name.size() - 1);
  }

  std::set<std::string> data_type;
  auto& phi_kernels = phi::KernelFactory::Instance().kernels();
  for (auto& kernel_pair : phi_kernels) {
    auto fluid_op_name = phi::TransToFluidOpName(kernel_pair.first);
    if (kernel_pair.first != op_name && fluid_op_name != op_name &&
        kernel_pair.first != non_inplace_op_name &&
        fluid_op_name != non_inplace_op_name) {
      continue;
    }
    for (auto& info_pair : kernel_pair.second) {
      data_type.insert(phi::DataTypeToString(info_pair.first.dtype()));
    }
  }

  return data_type;
}

void DoCheck(const pir::Value& value,
             const std::string& input_name,
             const std::set<std::string>& expected_dtype,
             const std::string& op_name) {
  if (value.type().isa<pir::DenseTensorType>()) {
    std::string value_type = phi::DataTypeToString(dialect::TransToPhiDataType(
        value.type().dyn_cast<pir::DenseTensorType>().dtype()));
    if (expected_dtype.find(value_type) == expected_dtype.end()) {
      std::ostringstream joined;
      std::copy(expected_dtype.begin(),
                expected_dtype.end(),
                std::ostream_iterator<std::string>(joined, ","));
      PADDLE_THROW(phi::errors::InvalidArgument(
          "Check data type error for op: %s, input: %s, %s.dtype is %s, and "
          "expected_dtype is %s",
          op_name,
          input_name,
          input_name,
          value_type,
          joined.str()));
    }
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get dtype for dense "
        "tensor."));
  }
}

void CheckValueDataType(const pir::Value& value,
                        const std::string& input_name,
                        const std::string& op_name) {
  VLOG(6) << "CheckValueDataType for " << op_name << ", input: " << input_name;
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);
  DoCheck(value, input_name, expected_dtype, op_name);
}

void CheckVectorOfValueDataType(const std::vector<pir::Value>& vector_value,
                                const std::string& input_name,
                                const std::string& op_name) {
  VLOG(6) << "CheckVectorOfValueDataType for " << op_name
          << ", input: " << input_name;
  std::set<std::string> expected_dtype = GetRegisterDataType(op_name);
  for (auto& value : vector_value) {
    DoCheck(value, input_name, expected_dtype, op_name);
  }
}

phi::DataType GetValueDataType(const pir::Value& value) {
  if (value.type().isa<DenseTensorType>()) {
    return paddle::dialect::TransToPhiDataType(
        value.type().dyn_cast<DenseTensorType>().dtype());
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Currently, we can only get dtype for dense "
        "tensor."));
  }
}

phi::DataType GetPromoteType(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_values_vector,
    const phi::DataType& amp_dtype) {
  auto dst_type = amp_dtype;
  // only consider the dtype of input(X).
  if (op_name == "batch_norm" || op_name == "layer_norm" ||
      op_name == "sync_batch_norm" ||
      op_name == "moving_average_abs_max_scale") {
    if (GetValueDataType(amp_values_vector[0][0]) == phi::DataType::FLOAT32) {
      dst_type = phi::DataType::FLOAT32;
    }
    return dst_type;
  }

  if (egr::Controller::Instance().GetCurrentTracer()->GetAmpDtype() ==
      "float16") {
    if (op_name == "fused_attention") {
      for (size_t i = 0; i < amp_values_vector.size(); i++) {
        if (i != 3 || i != 4 || i != 9 || i != 10) {
          if (GetValueDataType(amp_values_vector[i][0]) ==
              phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    } else if (op_name == "fused_feedforward") {
      for (size_t i = 0; i < amp_values_vector.size(); i++) {
        if (i != 7 || i != 8 || i != 9 || i != 10) {
          if (GetValueDataType(amp_values_vector[i][0]) ==
              phi::DataType::FLOAT32) {
            dst_type = phi::DataType::FLOAT32;
            return dst_type;
          }
        }
      }
    }
  }

  for (const auto& values : amp_values_vector) {
    for (const auto& value : values) {
      if (GetValueDataType(value) == phi::DataType::FLOAT32) {
        dst_type = GetValueDataType(value);
        break;
      }
    }
  }

  return dst_type;
}

pir::Value Cast(const pir::Value& input, const phi::DataType& dst_dtype) {
  paddle::imperative::AutoCastGuard guard(
      egr::Controller::Instance().GetCurrentTracer(),
      paddle::imperative::AmpLevel::O0);
  return paddle::dialect::cast(input, dst_dtype);
}

bool NeedCast(const pir::Value& value, const phi::DataType& dst_dtype) {
  auto data_type = GetValueDataType(value);
  if ((data_type == phi::DataType::FLOAT32 ||
       data_type == phi::DataType::FLOAT16 ||
       data_type == phi::DataType::BFLOAT16) &&
      (data_type != dst_dtype)) {
    return true;
  }
  return false;
}

pir::Value PirAmpAutoCast(const std::string& input_name,
                          const pir::Value& input,
                          const phi::DataType& dst_dtype,
                          const std::string& op_name) {
  VLOG(6) << "AMP AmpAutoCasts:"
          << " input(" << input_name << " to dst_dtype("
          << phi::DataTypeToString(dst_dtype) << ").";
  if ((op_name == "batch_norm" || op_name == "layer_norm" ||
       op_name == "sync_batch_norm" || op_name == "weight_only_linear") &&
      input_name != "x") {
    return input;
  }

  if (dst_dtype == phi::DataType::FLOAT16) {
    if (op_name == "run_program") {
      return input;
    }
    if ((op_name == "fused_attention" || op_name == "fused_feedforward")) {
      if (input_name == "LnScale" || input_name == "LnBias" ||
          input_name == "Ln2Scale" || input_name == "Ln2Bias" ||
          input_name == "Ln1Scale" || input_name == "Ln1Bias") {
        return input;
      }
    }
  }
  if (NeedCast(input, dst_dtype)) {
    VLOG(6) << "Input : " << input_name << "NeedCast";
    return Cast(input, dst_dtype);
  }
  return input;
}

phi::DataType GetAmpDestDtype(
    const std::string& op_name,
    const std::vector<std::vector<pir::Value>>& amp_values_vector) {
  auto amp_level = egr::Controller::Instance().GetAMPLevel();
  auto amp_setting_dtype =
      egr::Controller::Instance().GetCurrentTracer()->GetAmpPhiDtype();
  auto dst_type = amp_setting_dtype;

  bool use_promote = true;
  if (amp_level == paddle::imperative::AmpLevel::O2) {
    use_promote =
        egr::Controller::Instance().GetCurrentTracer()->GetUsePromote();
  }

  if (use_promote) {
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableAllowOps()
            ->count(op_name)) {
      dst_type = amp_setting_dtype;
    } else if (paddle::imperative::AmpOperators::Instance()
                   .GetMutableBlockOps()
                   ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    } else {
      if (amp_level == paddle::imperative::AmpLevel::OD) {
        dst_type = phi::DataType::FLOAT32;
      } else {
        dst_type =
            GetPromoteType(op_name, amp_values_vector, amp_setting_dtype);
      }
    }
  } else {
    // use_promote can be set to false only for O2 training.
    if (paddle::imperative::AmpOperators::Instance()
            .GetMutableBlockOps()
            ->count(op_name)) {
      dst_type = phi::DataType::FLOAT32;
    }
  }

  if (dst_type == amp_setting_dtype &&
      (paddle::imperative::AmpOperators::Instance()
           .GetMutableUnsupportedOps(amp_setting_dtype)
           ->count(op_name))) {
    dst_type = phi::DataType::FLOAT32;
  }

  // dst_type = GetDtypeWithPlace(op_name, amp_values_vector, dst_type);
  VLOG(6) << "AMP GetAmpDestDtype:"
          << " op(" << op_name << ") amp_dtype(" << dst_type << ") amp_level("
          << static_cast<int>(amp_level) << ").";
  return dst_type;
}

}  // namespace dialect
}  // namespace paddle

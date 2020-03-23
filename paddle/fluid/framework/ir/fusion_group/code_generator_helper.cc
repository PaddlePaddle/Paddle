/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/fusion_group/code_generator_helper.h"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/fusion_group/operation.h"
#include "paddle/fluid/framework/op_call_stack.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/shape_inference.h"
#include "paddle/fluid/framework/var_type_inference.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

template <typename T>
static T StringTo(const std::string& str) {
  std::istringstream is(str);
  T value;
  is >> value;
  return value;
}

static std::string ExpandMultivariateTemplate(const std::string& rhs,
                                              const size_t input_size) {
  int start_pos = rhs.find("[", 0);
  int end_pos = rhs.find("]", 0);
  std::string sum_rhs = rhs.substr(0, start_pos);
  std::string repeated_component =
      rhs.substr(start_pos + 1, (end_pos - start_pos - 1));
  int replace_pos = repeated_component.find("?", 0);

  for (size_t i = 1; i < input_size; i++) {
    std::string append_str = repeated_component;
    append_str.replace(replace_pos, 1, std::to_string(i));
    sum_rhs = sum_rhs + append_str;
  }
  return sum_rhs;
}

static std::string RefineTemplateWithAttr(const std::string& op_type,
                                          const std::string& attr_name,
                                          const AttributeMap& attrs) {
  std::string ret;
  // here str_cvt convert string to number in some attr
  // for example in fill_constant str_value
  std::stringstream str_cvt;
  auto IsNumber = [attr_name]() -> bool {
    return attr_name.find_first_not_of("0123456789") == std::string::npos;
  };

  if (!IsNumber()) {
    // Get attr with different type
    auto it = attrs.find(attr_name);
    Attribute attr = it->second;
    proto::AttrType attr_type = static_cast<proto::AttrType>(attr.which() - 1);
    if (attr_type == proto::AttrType::BOOLEANS) {
      bool result = boost::get<bool>(attr);
      if (result) {
        ret = "true";
      } else {
        ret = "false";
      }
    }
    if (attr_type == proto::AttrType::INTS) {
      int result = boost::get<int>(attr);
      str_cvt << result;
      ret = str_cvt.str();
    }
    if (attr_type == proto::AttrType::LONGS) {
      int64_t result = boost::get<int64_t>(attr);
      str_cvt << result;
      ret = str_cvt.str();
    }
    if (attr_type == proto::AttrType::FLOATS) {
      float result = boost::get<float>(attr);
      str_cvt << result;
      ret = str_cvt.str();
    }
    if (attr_type == proto::AttrType::STRINGS) {
      std::string result = boost::get<std::string>(attr);
      str_cvt << result;
      ret = str_cvt.str();
    }
  } else {
    ret = attr_name;
  }

  return ret;
}

// In order to avoid multiple __half2float function calls, we do this
// optimization
static std::string OptimzeFP16RHS(std::unordered_set<int>* used,
                                  const int index,
                                  const std::vector<int>& input_ids) {
  std::stringstream ret;
  if (used->find(input_ids[index]) == used->end()) {
    ret << "float half2fp32_" + TmpName(input_ids[index]) + " = __half2float(" +
               TmpName(input_ids[index]) + ");";
  }

  return ret.str();
}

std::string OperationExpression::GetRHS(std::unordered_set<int>* used,
                                        std::string* half2fp32_statement,
                                        size_t exprs_index) const {
  auto rhs = OperationMap::Instance().Get(op_type_).exprs[exprs_index];
  auto num_operands = OperationMap::Instance().Get(op_type_).num_operands;

  if (num_operands == -1) {
    size_t input_size = input_ids_.size();
    rhs = ExpandMultivariateTemplate(rhs, input_size);
  }
  for (size_t i = 0; i < rhs.size(); i++) {
    size_t pos = i;
    if (rhs[pos] == '$' && rhs[pos + 1] == '{') {
      int length = 0;
      while (rhs[pos + 2 + length] != '}') {
        length++;
      }
      std::string index_str = rhs.substr(pos + 2, length);
      std::string refine_str =
          RefineTemplateWithAttr(op_type_, index_str, attr_);
      if (index_str == refine_str) {
        int index = StringTo<int>(index_str);
        PADDLE_ENFORCE_LT(index, input_ids_.size(),
                          platform::errors::InvalidArgument(
                              "Only %d inputs are provided, but need %d for "
                              "operation < %s >.",
                              input_ids_.size(), index + 1, op_type_));
        PADDLE_ENFORCE_GE(input_ids_[index], 0,
                          platform::errors::InvalidArgument(
                              "Expected %d-th input id > 0 for operation < %s "
                              ">. Received %d.",
                              index, op_type_, input_ids_[index]));
        // TODO(wangchaochaohu): Here fp16 convert to float to do comupte, we
        // need
        // to add general fp16 compute later.
        std::string var_name;
        if (rhs_type_ == "float16") {
          half2fp32_statement->append(OptimzeFP16RHS(used, index, input_ids_));
          var_name = "half2fp32_" + TmpName(input_ids_[index]);
        } else {
          var_name = TmpName(input_ids_[index]);
        }
        rhs.replace(pos, length + 3, var_name);
        used->insert(input_ids_[index]);
      }
    }
  }
  return rhs;
}

std::string OperationExpression::GetLHS(size_t i) const {
  std::stringstream ret;
  ret << lhs_type_ << " " << TmpName(output_ids_[i]);
  return ret.str();
}

bool OperationExpression::IsSupport() const {
  return OperationMap::Instance().Has(op_type_);
}

// we Traverse the graph and get the group , all input id and output id is
// unique for the node which belong the group
std::string OperationExpression::GetExpression(
    std::unordered_set<int>* used) const {
  std::string half2fp32_statement;
  std::stringstream ret;
  if (IsSupport()) {
    for (size_t i = 0; i < output_ids_.size(); ++i) {
      std::string cast_str = "";
      if ((lhs_type_ == rhs_type_ && rhs_type_ != "float16") ||
          (lhs_type_ != rhs_type_ && rhs_type_ == "float16")) {
        ret << GetLHS(i) << " = " << GetRHS(used, &half2fp32_statement, i)
            << ";";
      } else {
        if ((lhs_type_ == rhs_type_ && rhs_type_ == "float16") ||
            lhs_type_ == "float16") {
          cast_str = "__float2half";
        } else {
          cast_str = "static_cast<" + lhs_type_ + ">";
        }
        ret << GetLHS(i) << " = " << cast_str << "("
            << GetRHS(used, &half2fp32_statement, i) << ");";
      }
    }
  }
  return half2fp32_statement + ret.str();
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle

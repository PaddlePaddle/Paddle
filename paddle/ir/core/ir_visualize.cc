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

#include <list>
#include <ostream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/dot_lang.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/value.h"

namespace ir {
size_t cur_var_number_{0};
std::unordered_map<const void*, std::string> aliases_;

inline std::string GenNodeId(const Value& v) {
  const void* key = static_cast<const void*>(v.impl());
  auto ret = aliases_.find(key);
  if (ret != aliases_.end()) {
    return ret->second;
  }

  std::string id = std::to_string(cur_var_number_);
  cur_var_number_++;
  aliases_[key] = id;
  return id;
}

inline std::string GetAttribute(Attribute attr) {
  if (!attr) {
    return "";
  }

  if (auto s = attr.dyn_cast<StrAttribute>()) {
    return s.AsString();
  } else if (auto b = attr.dyn_cast<BoolAttribute>()) {
    return std::to_string(b.data());
  } else if (auto f = attr.dyn_cast<FloatAttribute>()) {
    return std::to_string(f.data());
  } else if (auto d = attr.dyn_cast<DoubleAttribute>()) {
    return std::to_string(d.data());
  } else if (auto i = attr.dyn_cast<Int32Attribute>()) {
    return std::to_string(i.data());
  } else if (auto i = attr.dyn_cast<Int64Attribute>()) {
    return std::to_string(i.data());
  } else if (auto arr = attr.dyn_cast<ArrayAttribute>()) {
    const auto& vec = arr.AsVector();
    return "array";
  }
  return "";
}

inline std::string GetType(Type type) {
  if (!type) {
    return "";
  }

  std::string res_type = type.dialect().name() + ".";
  if (auto tensor_type = type.dyn_cast<DenseTensorType>()) {
    res_type += "tensor<";
    for (auto d : phi::vectorize(tensor_type.dims())) {
      res_type += std::to_string(d) + "x";
    }

    if (tensor_type.dtype().isa<BFloat16Type>()) {
      return res_type + "bf16" + ">";
    } else if (tensor_type.dtype().isa<Float16Type>()) {
      return res_type + "f16" + ">";
    } else if (tensor_type.dtype().isa<Float32Type>()) {
      return res_type + "f32" + ">";
    } else if (tensor_type.dtype().isa<Float64Type>()) {
      return res_type + "f64" + ">";
    } else if (tensor_type.dtype().isa<BoolType>()) {
      return res_type + "b" + ">";
    } else if (tensor_type.dtype().isa<Int8Type>()) {
      return res_type + "i8" + ">";
    } else if (tensor_type.dtype().isa<UInt8Type>()) {
      return res_type + "u8" + ">";
    } else if (tensor_type.dtype().isa<Int16Type>()) {
      return res_type + "i16" + ">";
    } else if (tensor_type.dtype().isa<Int32Type>()) {
      return res_type + "i32" + ">";
    } else if (tensor_type.dtype().isa<Int64Type>()) {
      return res_type + "i64" + ">";
    } else if (tensor_type.dtype().isa<IndexType>()) {
      return res_type + "index" + ">";
    } else if (tensor_type.dtype().isa<Complex64Type>()) {
      return res_type + "c64" + ">";
    } else if (tensor_type.dtype().isa<Complex128Type>()) {
      return res_type + "c128" + ">";
    } else if (tensor_type.dtype().isa<VectorType>()) {
      return res_type + "vec" + ">";
    } else {
      return "";
    }
  } else {
    return "";
  }
}

inline std::string GenNodeLabel(const Operation* op,
                                std::string id,
                                bool is_op = false) {
  std::string label = "";

  if (!is_op) {
    label += "Var_" + id + "\\n";
    // get Op attributes
    AttributeMap attributes = op->attributes();
    std::map<std::string, Attribute, std::less<std::string>> order_attributes(
        attributes.begin(), attributes.end());
    for (const auto& it : order_attributes) {
      label += it.first + ":" + GetAttribute(it.second) + "\\n";
    }

    // get Op Type
    std::vector<Type> op_result_types;
    auto num_op_result = op->num_results();
    op_result_types.reserve(num_op_result);
    for (size_t idx = 0; idx < num_op_result; idx++) {
      auto op_result = op->result(idx);
      if (op_result) {
        label += GetType(op_result.type()) + "\\n";
      }
    }
  } else {
    label += op->name() + "_" + id;
  }

  return label;
}

inline std::vector<utils::DotAttr> GenNodeAttr(bool is_op = false) {
  if (!is_op) {
    return std::vector<utils::DotAttr>{utils::DotAttr("color", "#FFDC85"),
                                       utils::DotAttr("style", "filled")};
  } else {
    return std::vector<utils::DotAttr>{utils::DotAttr("shape", "Mrecord"),
                                       utils::DotAttr("color", "#8EABFF"),
                                       utils::DotAttr("style", "filled")};
  }
}

void Program::Visualize() const {
  auto top_level_op = this->module_op();
  auto& region = top_level_op->region(0);
  auto block = region.front();
  utils::DotLang dot;

  for (auto op : *block) {
    // Get Node id
    auto num_op_result = op->num_results();
    std::vector<OpResult> op_results;
    op_results.reserve(num_op_result);
    for (size_t idx = 0; idx < num_op_result; idx++) {
      op_results.push_back(op->result(idx));
    }
    std::string id = "";
    for (const auto& value : op_results) {
      id = GenNodeId(value);
    }

    // Add Node
    auto num_op_operands = op->num_operands();

    if (num_op_result > 0) {
      std::string var_id = "var_" + id;
      std::string param_node_label = GenNodeLabel(op, id);
      std::vector<utils::DotAttr> param_node_attr = GenNodeAttr();
      dot.AddNode(var_id, param_node_attr, param_node_label);
      if (num_op_operands > 0) {
        std::string op_id = "op_" + id;
        std::string op_node_label = GenNodeLabel(op, id, true);
        std::vector<utils::DotAttr> param_node_attr = GenNodeAttr(true);
        dot.AddNode(op_id, param_node_attr, op_node_label);

        // Add Edge
        std::vector<Value> op_operands;
        op_operands.reserve(num_op_operands);
        for (size_t idx = 0; idx < num_op_operands; idx++) {
          op_operands.push_back(op->operand_source(idx));
        }
        for (const auto& value : op_operands) {
          std::string source_id = "var_" + GenNodeId(value);
          dot.AddEdge(source_id, op_id, {});
        }
        dot.AddEdge(op_id, var_id, {});
      }
    }
  }
  std::cout << dot() << std::endl;
}
}  // namespace ir

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

#pragma once

#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
namespace fusion_group {

static inline std::string ArgName(int index) {
  return "arg" + std::to_string(index);
}

static inline std::string TmpName(int index) {
  return "tmp" + std::to_string(index);
}

static inline std::string VarName(int index) {
  return "arg" + std::to_string(index) + "[idx]";
}

class OperationExpression {
 public:
  explicit OperationExpression(
      std::string op_type,
      const std::vector<int>& input_ids,
      const std::vector<int>& output_ids,
      std::string rhs_type,
      std::string lhs_type,
      const std::vector<int>& intermediate_output_ids = {})
      : op_type_(op_type),
        input_ids_(input_ids),
        output_ids_(output_ids),
        rhs_type_(rhs_type),
        lhs_type_(lhs_type),
        intermediate_output_ids_(intermediate_output_ids) {}

  std::string GetOpType() const { return op_type_; }
  std::vector<int> GetInputIds() const { return input_ids_; }
  std::vector<int> GetOutputIds() const { return output_ids_; }
  std::vector<int> GetIntermediateOutputIds() const {
    return intermediate_output_ids_;
  }
  std::string GetRHSType() const { return rhs_type_; }
  std::string GetLHSType() const { return lhs_type_; }
  void SetAttr(AttributeMap attr) { attr_ = attr; }
  AttributeMap GetAttr() { return attr_; }
  // Check whether this operation type is supported in OperationMap.
  bool IsSupport() const;

  std::string GetExpression(std::unordered_set<int>* used) const;

 private:
  // TODO(wangchao): make offset more flexible we add stride and basic offset
  std::string GetRHS(std::unordered_set<int>* used,
                     size_t exprs_index = 0) const;
  std::string GetLHS(size_t i = 0) const;

 private:
  std::string op_type_;
  std::vector<int> input_ids_;
  std::vector<int> output_ids_;
  AttributeMap attr_;
  std::string rhs_type_;
  std::string lhs_type_;
  std::vector<int> intermediate_output_ids_;
};

class TemplateVariable {
 public:
  void Add(std::string identifier, std::string expression) {
    strings_[identifier] = expression;
  }

  void Remove(std::string identifier, std::string expression UNUSED) {
    for (auto it = strings_.begin(); it != strings_.end();) {
      if (it->first == identifier) {
        it = strings_.erase(it);
      } else {
        it++;
      }
    }
  }

  std::unordered_map<std::string, std::string> Get() { return strings_; }

 private:
  std::unordered_map<std::string, std::string> strings_;
};

class CodeTemplate {
 public:
  CodeTemplate() = default;
  explicit CodeTemplate(std::string template_str) {
    template_str_ = template_str;
  }

  std::string Format(TemplateVariable template_var) {
    std::string ret = template_str_;
    std::unordered_map<std::string, bool> found;

    // Word begins with "$" in template_str will be replaced.
    for (size_t i = 0; i < ret.size(); i++) {
      auto pos = i;
      char c = ret[pos];

      if (c == '$') {
        for (auto iter : template_var.Get()) {
          std::string keyword = iter.first;
          if (ret.substr(pos + 1, keyword.size()) == keyword) {
            found[keyword] = true;
            ret.replace(pos, keyword.size() + 1, iter.second);
            break;
          }
        }
      }
    }

    for (auto iter : template_var.Get()) {
      PADDLE_ENFORCE_NE(found.find(iter.first),
                        found.end(),
                        common::errors::PreconditionNotMet(
                            "Keyword %s in template is not set.", iter.first));
    }

    return EmitIndents(ret);
  }

  std::string EmitIndents(std::string str) {
    std::string ret = str;
    int space_num = 0;
    auto space_char = ' ';
    for (size_t i = 0; i < ret.size(); i++) {
      auto pos = i;
      char c = ret[pos];
      if (c == '\n') {
        size_t next_pos = pos + 1;
        while (next_pos < ret.size() && ret[next_pos] == space_char) {
          next_pos++;
        }
        space_num = next_pos - pos - 1;
      }
      if (c == ';' && (pos + 1 < ret.size()) && ret[pos + 1] != '\n') {
        auto insert_pos = pos + 1;
        std::string insert_str = "\n" + std::string(space_num, space_char);
        ret.insert(insert_pos, insert_str);
        space_num = 0;
      }
    }

    return ret;
  }

 private:
  std::string template_str_;
};

static std::string DebugString(const OperationExpression& expr) {
  std::stringstream ret;
  ret << "Op(" << expr.GetOpType() << "), inputs:{";
  auto input_ids = expr.GetInputIds();
  for (size_t i = 0; i < input_ids.size(); ++i) {
    if (i != 0) {
      ret << ",";
    }
    ret << expr.GetInputIds()[i];
  }
  ret << "}, outputs:{";
  auto output_ids = expr.GetOutputIds();
  for (size_t i = 0; i < output_ids.size(); ++i) {
    if (i != 0) {
      ret << ",";
    }
    ret << expr.GetOutputIds()[i];
  }
  ret << "}";
  return ret.str();
}

}  // namespace fusion_group
}  // namespace ir
}  // namespace framework
}  // namespace paddle

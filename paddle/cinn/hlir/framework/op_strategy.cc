// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/hlir/framework/op_strategy.h"

#include "paddle/common/errors.h"

namespace {

struct PyBindNodeAttrVisitor {
  std::stringstream& out;
  explicit PyBindNodeAttrVisitor(std::stringstream& out) : out(out) {}

  void operator()(int v) { out << "int: " << v; }
  void operator()(int64_t v) { out << "int64_t: " << v; }
  void operator()(float v) { out << "float: " << v; }
  void operator()(double v) { out << "double: " << v; }
  void operator()(bool v) { out << "bool: " << v; }
  void operator()(const std::string& v) { out << "string: " << v; }
#define VISIT_ELEMENTS(T__)                                      \
  void operator()(const std::vector<T__>& vs) {                  \
    if (vs.empty()) return;                                      \
    for (int i = 0; i < vs.size() - 1; i++) out << vs[i] << ","; \
    out << vs.back();                                            \
  }
  VISIT_ELEMENTS(int)
  VISIT_ELEMENTS(int64_t)
  VISIT_ELEMENTS(float)
  VISIT_ELEMENTS(double)
  VISIT_ELEMENTS(bool)
  VISIT_ELEMENTS(std::string)
  VISIT_ELEMENTS(symbol::DimExpr)
  VISIT_ELEMENTS(cinn::dialect::SymbolBinding)
};

}  // namespace

namespace cinn {
namespace hlir {
namespace framework {

std::ostream& operator<<(std::ostream& os, const NodeAttr& node_attr) {
  std::stringstream ss;
  ss << "NodeAttr:\n";
  for (auto& item : node_attr.attr_store) {
    std::stringstream os;
    PyBindNodeAttrVisitor visitor(os);
    absl::visit(visitor, item.second);
    ss << "- " << os.str() << "\n";
  }
  os << ss.str();
  return os;
}

std::shared_ptr<OpImpl> OpStrategy::SelectImpl(
    const std::shared_ptr<OpStrategy>& strategy) {
  //! should get the host info from global environment.
  std::string curr_condition = "default";
  std::shared_ptr<OpImpl> res = nullptr;
  for (auto& spec : strategy->specializations) {
    if (spec->condition == "default") {
      for (auto& i : spec->implementations) {
        if (!res || res->plevel < i->plevel) {
          res = i;
        }
      }
    }
  }
  PADDLE_ENFORCE_NE(
      res,
      nullptr,
      ::common::errors::NotFound(
          "There is no available strategy implementation! SelectImpl failed!"));
  return res;
}

void OpStrategy::AddImpl(CINNCompute fcompute,
                         CINNSchedule fschedule,
                         std::string name,
                         int plevel) {
  //! TODO(haozech) : here curr_cond should get the condition from outside.
  //! Expected : auto curr_cond = SpecializedCondition::Current();
  std::string curr_condition = "default";
  for (auto& op_spec : specializations) {
    if (op_spec->condition == curr_condition) {
      op_spec->AddImpl(fcompute, fschedule, std::move(name), plevel);
      return;
    }
  }
  std::shared_ptr<OpSpec> n = std::make_shared<OpSpec>();
  n->condition = curr_condition;
  n->AddImpl(fcompute, fschedule, std::move(name), plevel);
  this->specializations.push_back(n);
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

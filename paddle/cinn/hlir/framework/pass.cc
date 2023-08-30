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

#include "paddle/cinn/hlir/framework/pass.h"

#include "paddle/cinn/hlir/framework/visualize_helper.h"
#include "paddle/cinn/hlir/pass/use_pass.h"

namespace cinn {
namespace hlir {
namespace framework {

void ApplyPasses(Graph* g, const std::vector<std::string>& passes) {
  std::vector<const PassFunctionRegister*> fpass;
  for (auto& name : passes) {
    VLOG(1) << "Run Pass -> " << name;
    auto* reg = Registry<PassFunctionRegister>::Global()->Find(name);
    CHECK(reg) << "Cannot find pass " << name << " in the registry";
    fpass.push_back(reg);
  }
  for (auto* r : fpass) {
    cinn::hlir::framework::PassPrinter::GetInstance()->PassBegin(r->name, g);
    for (auto& dep : r->graph_attr_dependency) {
      CHECK_NE(g->attrs.count(dep), 0)
          << "To apply pass [" << r->name << "], Graph's attribute [" << dep
          << "] is required, but it is not available.";
      if (g->attrs.count(dep) == 0) {
        auto* pass_dep = FindPassDep(dep);
        CHECK(!pass_dep) << "And the attribute is provided by pass ["
                         << pass_dep->name << "].";
      }
    }
    r->body(g);
    cinn::hlir::framework::PassPrinter::GetInstance()->PassEnd(r->name, g);
  }
}

const PassFunctionRegister* FindPassDep(const std::string& attr_name) {
  for (auto* r : Registry<PassFunctionRegister>::Global()->List()) {
    for (auto& s : r->graph_attr_targets) {
      if (s == attr_name) return r;
    }
  }
  return nullptr;
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn

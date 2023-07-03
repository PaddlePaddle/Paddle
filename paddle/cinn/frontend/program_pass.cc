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

#include "paddle/cinn/frontend/program_pass.h"

#include <unordered_set>

#include "paddle/cinn/hlir/framework/visualize_helper.h"

namespace cinn {
namespace frontend {

void ProgramPass::Apply(Program* prog,
                        const std::unordered_set<std::string>& fetch_ids,
                        const common::Target& target,
                        const std::vector<std::string>& passes) {
  std::vector<const ProgramPass*> fpass;
  for (auto& name : passes) {
    const auto* pass = ProgramPassRegistry::Global()->Get(name);
    fpass.push_back(pass);
  }
  for (const auto* pass : fpass) {
    int before = prog->size();
    cinn::hlir::framework::PassPrinter::GetInstance()->PassBegin(pass->name(),
                                                                 *prog);
    pass->ApplyImpl(prog, fetch_ids, target);
    const_cast<ProgramPass*>(pass)->Clear();
    int after = prog->size();
    cinn::hlir::framework::PassPrinter::GetInstance()->PassEnd(pass->name(),
                                                               *prog);
    VLOG(1) << "Apply " << pass->name() << " pass, program size: " << before
            << " -> " << after << ", diff: " << after - before;
  }
}

}  // namespace frontend
}  // namespace cinn

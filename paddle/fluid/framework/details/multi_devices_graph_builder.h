//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace platform {
class NCCLContextMap;
}

namespace framework {
class Scope;
namespace details {
class MultiDevSSAGraphBuilder : public SSAGraphBuilder {
 public:
  MultiDevSSAGraphBuilder(const std::vector<platform::Place> &places,
                          const std::string &loss_var_name,
                          const std::unordered_set<std::string> &params,
                          const std::vector<Scope *> &local_scopes,
                          platform::NCCLContextMap *nccl_ctxs);

  std::unique_ptr<SSAGraph> Build(const ProgramDesc &program) const override;

 private:
  std::string loss_var_name_;
  const std::vector<platform::Place> &places_;
  const std::vector<Scope *> &local_scopes_;
  platform::NCCLContextMap *nccl_ctxs_;
  std::unordered_set<std::string> grad_names_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle

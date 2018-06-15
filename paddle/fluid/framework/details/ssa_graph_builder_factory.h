// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/ssa_graph_builder.h"
#include "paddle/fluid/platform/place.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {
class Scope;
namespace details {

class SSAGraphBuilderFactory {
 public:
  SSAGraphBuilderFactory(const std::vector<platform::Place>& places,
                         const std::string& loss_var_name,
                         const std::unordered_set<std::string>& param_names,
                         const std::vector<Scope*>& local_scopes,
                         const BuildStrategy& strategy)
      : places_(places),
        loss_var_name_(loss_var_name),
        param_names_(param_names),
        local_scopes_(local_scopes),
        strategy_(strategy) {
#ifdef PADDLE_WITH_CUDA
    nccl_ctxs_ = nullptr;
#endif
  }

#ifdef PADDLE_WITH_CUDA
  void SetNCCLContextMap(platform::NCCLContextMap* nccl_ctxs) {
    nccl_ctxs_ = nccl_ctxs;
  }
#endif

  std::unique_ptr<SSAGraphBuilder> Create();

 private:
  std::vector<platform::Place> places_;
  std::string loss_var_name_;
  std::unordered_set<std::string> param_names_;
  std::vector<Scope*> local_scopes_;
  BuildStrategy strategy_;

#ifdef PADDLE_WITH_CUDA
  platform::NCCLContextMap* nccl_ctxs_;
#endif
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

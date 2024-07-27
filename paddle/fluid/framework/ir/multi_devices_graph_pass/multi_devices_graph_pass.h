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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/build_strategy.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/ir/graph.h"

namespace paddle {
namespace framework {
namespace details {
class OpHandleBase;
struct VarHandle;
}  // namespace details
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace platform {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
class NCCLCommunicator;
class NCCLContextMap;
#elif defined(PADDLE_WITH_XPU_BKCL)
class BKCLContextMap;
class BKCLCommunicator;
#endif
}  // namespace platform

namespace framework {
class Scope;

namespace ir {

constexpr char kLossVarName[] = "loss_var_name";
constexpr char kStrategy[] = "strategy";

class MultiDevSSAGraphBuilderBase : public ir::Pass {
 protected:
  void ApplyImpl(ir::Graph *graph) const override;

  virtual void Init() const;

  virtual void CheckGraph(const ir::Graph &graph) const;

  virtual std::vector<ir::Node *> SortOperations(const ir::Graph &graph) const;

  void CreateComputationalOps(ir::Graph *result,
                              ir::Node *node,
                              size_t num_places) const;

  void CreateScaleLossGradOp(ir::Graph *result,
                             const std::string &loss_grad_name,
                             ir::Node *out_var_node,
                             size_t loss_scale,
                             proto::VarType::Type dtype) const;

  void CreateComputationalOp(ir::Graph *result,
                             ir::Node *node,
                             size_t dev_id) const;

  void CreateOpHandleIOs(ir::Graph *result,
                         ir::Node *node,
                         size_t device_id) const;

  void CreateIsolatedVarNode(ir::Graph *result, ir::Node *var_node) const;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
  mutable platform::NCCLContextMap *nccl_ctxs_{nullptr};
  mutable platform::NCCLCommunicator *multi_nccl_ctxs_{nullptr};
#elif defined(PADDLE_WITH_XPU_BKCL)
  mutable platform::BKCLContextMap *bkcl_ctxs_{nullptr};
  mutable platform::BKCLCommunicator *multi_bkcl_ctxs_{nullptr};
#endif

  mutable std::string loss_var_name_;
  mutable std::vector<phi::Place> places_;
  mutable std::vector<Scope *> local_scopes_;

  mutable details::BuildStrategy strategy_;
  mutable std::unordered_map<std::string, VarDesc *> all_vars_;
};

std::unordered_set<std::string> &MultiDevSSAGraphBuilder();

}  // namespace ir
}  // namespace framework
}  // namespace paddle

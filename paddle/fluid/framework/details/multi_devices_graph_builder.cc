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

#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/scale_loss_grad_op_handle.h"
#include "paddle/fluid/framework/details/send_op_handle.h"
#include "paddle/fluid/framework/scope.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/details/nccl_all_reduce_op_handle.h"
#endif

#include <string>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

#ifdef PADDLE_WITH_CUDA
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes,
    platform::NCCLContextMap *nccl_ctxs, bool distributed)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      distributed_(distributed),
      nccl_ctxs_(nccl_ctxs) {
#else
MultiDevSSAGraphBuilder::MultiDevSSAGraphBuilder(
    const std::vector<platform::Place> &places,
    const std::string &loss_var_name,
    const std::unordered_set<std::string> &params,
    const std::vector<Scope *> &local_scopes, bool distributed)
    : loss_var_name_(loss_var_name),
      places_(places),
      local_scopes_(local_scopes),
      distributed_(distributed) {
#endif
  for (auto &p : params) {
    grad_names_.insert(GradVarName(p));
  }
}

void MultiDevSSAGraphBuilder::CreateOpHandleIOs(SSAGraph *result, OpDesc *op,
                                                const platform::Place &p,
                                                const size_t &i) const {
  auto *op_handle = result->ops_.back().get();

  auto var_names = op->InputArgumentNames();

  for (auto &each_var_name : var_names) {
    VarHandle *var = CreateOrGetLatestVarHandle(result, each_var_name, p, i);
    op_handle->AddInput(var);
  }
  var_names = op->OutputArgumentNames();

  for (auto &each_var_name : var_names) {
    CreateOpOutput(result, op_handle, each_var_name, p, i);
  }
}

std::unique_ptr<SSAGraph> MultiDevSSAGraphBuilder::Build(
    const ProgramDesc &program) const {
  auto graph = new SSAGraph();
  SSAGraph &result = *graph;
  std::unordered_set<std::string> og_has_been_broadcast;

  // We cannot invoke resize. It is a bug of GCC 4.8
  result.vars_ = std::vector<
      std::unordered_map<std::string, std::vector<std::unique_ptr<VarHandle>>>>(
      places_.size());

  bool is_forwarding = true;
  for (auto *op : program.Block(0).AllOps()) {
    bool change_forward = false;
    if (!is_forwarding) {
      // FIXME(yy): Do not hard code like this
      if (op->OutputArgumentNames().size() == 1 &&
          op->OutputArgumentNames()[0] == GradVarName(loss_var_name_)) {
        continue;  // Drop fill 1. for backward coeff;
      }
    }

    // append send op if program is distributed trainer main program.
    // always use the first device
    if (is_forwarding && distributed_ && op->Type() == "send") {
      auto &p = places_[0];
      auto *s = local_scopes_[0];
      size_t i = 0;
      result.ops_.emplace_back(new SendOpHandle(*op, s, p));
      CreateOpHandleIOs(&result, op, p, i);
      continue;
    }

    for (size_t i = 0; i < places_.size(); ++i) {
      auto &p = places_[i];
      auto *s = local_scopes_[i];

      result.ops_.emplace_back(new ComputationOpHandle(*op, s, p));
      auto *op_handle = result.ops_.back().get();
      op_handle->dev_ctxes_[p] = const_cast<platform::DeviceContext *>(
          platform::DeviceContextPool::Instance().Get(p));

      CreateOpHandleIOs(&result, op, p, i);
      // auto var_names = op->InputArgumentNames();

      // for (auto &each_var_name : var_names) {
      //   VarHandle *var =
      //       CreateOrGetLatestVarHandle(&result, each_var_name, p, i);
      //   op_handle->AddInput(var);
      // }
      auto var_names = op->OutputArgumentNames();

      // for (auto &each_var_name : var_names) {
      //   CreateOpOutput(&result, op_handle, each_var_name, p, i);
      // }

      if (is_forwarding) {
        if (var_names.size() == 1 && var_names[0] == loss_var_name_) {
// Insert ScaleCost OpHandle
#ifdef PADDLE_WITH_CUDA
          auto *communication_dev_ctx = nccl_ctxs_->DevCtx(p);
#else
          auto *communication_dev_ctx =
              platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
#endif

          op_handle = new ScaleLossGradOpHandle(local_scopes_.size(), s, p,
                                                communication_dev_ctx);
          result.ops_.emplace_back(op_handle);

          // FIXME: Currently ScaleLossGradOp only use device_count as scale
          // factor. So it does not depend on any other operators.
          // VarHandle *loss = GetVarHandle(loss_var_name, place);
          // loss->pending_ops_.emplace_back(op_handle);
          // op_handle->inputs_.emplace_back(loss);

          CreateOpOutput(&result, op_handle, GradVarName(loss_var_name_), p, i);
          change_forward = true;
        }
      }
    }

    if (change_forward) {
      is_forwarding = false;
    }

    if (!is_forwarding) {
      auto var_names = op->OutputArgumentNames();
      // Currently, we assume that once gradient is generated, it can be
      // broadcast, and each gradient is only broadcast once. But there are no
      // other cases, for example, we need to adjust the gradient according to
      // the input when we get the gradient, which is not considered at present.
      for (auto &og : var_names) {
        if (grad_names_.count(og) != 0 &&
            og_has_been_broadcast.count(og) == 0) {  // is param grad
                                                     // Insert NCCL AllReduce Op
          og_has_been_broadcast.insert(og);
#ifdef PADDLE_WITH_CUDA
          result.ops_.emplace_back(
              new NCCLAllReduceOpHandle(local_scopes_, places_, *nccl_ctxs_));
          auto *op_handle = result.ops_.back().get();

          for (size_t i = 0; i < places_.size(); ++i) {
            auto &p = places_[i];
            auto &vars = result.vars_[i][og];

            if (vars.empty()) {  // This device has no data. continue.
              continue;
            }
            auto &prev_grad = vars[vars.size() - 1];
            op_handle->AddInput(prev_grad.get());

            vars.emplace_back(new VarHandle);
            auto &var = vars.back();
            var->place_ = p;
            var->name_ = og;
            var->version_ = vars.size() - 1;

            op_handle->AddOutput(var.get());
          }
#else
          PADDLE_ENFORCE("Not implemented");
#endif
        }
      }
    }
  }

  /*
    Dependency graph has been constructed. However, there are still data
    harzaeds need to be handled.
   */
  PolishGraphToSupportDataHazards(&result);

  /*
   * Only variables should be the leaves of graph.
   */
  AddOutputToLeafOps(&result);

  if (VLOG_IS_ON(10)) {
    std::ostringstream sout;
    PrintGraphviz(*graph, sout);
    VLOG(10) << sout.str();
  }

  return std::unique_ptr<SSAGraph>(graph);
}  // namespace details
}  // namespace details
}  // namespace framework
}  // namespace paddle

/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <paddle/framework/backward.h>
#include <paddle/framework/net.h>

namespace paddle {
namespace framework {

static bool AllInSet(const std::vector<std::string>& names,
                     const std::string& suffix,
                     const std::unordered_set<std::string>& set) {
  for (auto& name : names) {
    if (set.find(name + suffix) == set.end()) {
      return false;
    }
  }
  return true;
}

static std::vector<size_t> InSetIdx(
    const std::vector<std::string>& names, const std::string& suffix,
    const std::unordered_set<std::string>& set) {
  std::vector<size_t> ret_val;
  ret_val.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    if (set.find(names[i] + suffix) != set.end()) {
      ret_val.push_back(i);
    }
  }
  return ret_val;
}

static std::shared_ptr<OperatorBase> EmptyOp() {
  auto net_op = std::make_shared<NetOp>();
  net_op->CompleteAddOp();
  return net_op;
}

static void DeDuplicate(NetOp* net, std::unordered_se)

    static std::shared_ptr<OperatorBase> BackwardImpl(
        const OperatorBase& forwardOp,
        std::unordered_set<std::string>& no_grad_names, unsigned& uniq_id) {
  if (AllInSet(forwardOp.inputs_, OperatorBase::GRAD_VAR_SUFFIX(),
               no_grad_names)) {
    return EmptyOp();
  }

  if (AllInSet(forwardOp.outputs_, OperatorBase::GRAD_VAR_SUFFIX(),
               no_grad_names)) {
    for (auto& name : forwardOp.inputs_) {
      // Mark all input is not need
      no_grad_names.insert(name + OperatorBase::GRAD_VAR_SUFFIX());
    }
    return EmptyOp();
  }

  auto* net = new NetOp();

  if (forwardOp.IsNetOp()) {
    std::unordered_map<std::string, int> dup_output;
    std::unordered_map<std::string std::vector<int>> dup_output_ops;
    const unsigned uniq_id_local = uniq_id;
    unsigned op_id_offset = 0;
    for (auto& fwd : forwardOp) {
      auto bwd = Backward(fwd, no_grad_names);
      net->AddOp(bwd);
      for (size_t i = 0; i < bwd.outputs_; ++i) {
        bwd->outputs_[i] += OperatorBase::EMPTY_VAR_NAME();
        if (dup_output.find(bwd->inputs_[i]) == dup_output.end()) {
          dup_output[bwd->inputs_[i]] = 1;
          dup_output_ops[bwd->inputs_[i]] = std::vector<int>{op_id_offset++};
        } else {
          dup_output[bwd->inputs_[i]]++;
          dup_output_ops[bwd->inputs_[i]].emplace_back(op_id_offset++);
        }
      }
    }
    for (auto dup : dup_output) {
      if (dup.second == 1) continue;
      auto op_ids = dup_output_ops.at(dup.first);
      for (auto& op_id : op_ids) {
        auto& op_ptr = net->ops_[op_id];
        for (size_t i = 0; i < op_ptr->inputs_.size(); ++i) {
          if (op_ptr->inputs_[i] == dup.first) {
            // unique the duplicate name
            op_ptr->inputs_[i] += std::to_string(uniq_id++);
          }
        }
      }
    }

    //! TODO(dzh)
  } else {
    //! TODO(fjy)
  }

  net->CompleteAddOp();
  return std::shared_ptr<OperatorBase>(net);
}

extern std::shared_ptr<OperatorBase> Backward(
    const OperatorBase& forwardOp,
    const std::unordered_set<std::string>& no_grad_vars) {
  std::unordered_set<std::string> no_grad_names;
  no_grad_names.reserve(no_grad_vars.size());

  for (auto& name : no_grad_vars) {
    no_grad_names.insert(name + OperatorBase::GRAD_VAR_SUFFIX());
  }
  int uid = 0;
  return BackwardImpl(forwardOp, no_grad_names, uid);
}
}  // namespace framework
}  // namespace paddle

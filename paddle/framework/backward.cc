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

static std::vector<int> InSetIdx(const std::vector<std::string>& names,
                                 const std::string& suffix,
                                 const std::unordered_set<std::string>& set) {
  std::vector<int> ret_val;
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

static std::shared_ptr<OperatorBase> BackwardImpl(
    const OperatorBase& forwardOp,
    std::unordered_set<std::string>& no_grad_names, int& uniq_id) {
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
    //! TODO(dzh)
  } else {
    //! TODO(fjy)
  }

  net->CompleteAddOp();
  return std::shared_ptr<OperatorBase>(net);
}

extern std::shared_ptr<OperatorBase> Backward(
    const std::shared_ptr<OperatorBase>& forwardOp,
    const std::unordered_set<std::string>& no_grad_vars) {
  std::unordered_set<std::string> no_grad_names;
  no_grad_names.reserve(no_grad_vars.size());

  for (auto& name : no_grad_vars) {
    no_grad_names.insert(name + OperatorBase::GRAD_VAR_SUFFIX());
  }
  int uid = 0;
  return BackwardImpl(*forwardOp, no_grad_names, uid);
}
}  // namespace framework
}  // namespace paddle

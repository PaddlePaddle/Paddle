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

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/all_reduce_op_handle.h"
#include "paddle/fluid/framework/details/dgc_const_values.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
namespace platform {
class NCCLCommunicator;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

class SparseAllReduceOpHandle : public AllReduceOpHandle {
 public:
  SparseAllReduceOpHandle(ir::Node *node,
                          const std::vector<Scope *> &local_scopes,
                          const std::vector<platform::Place> &places,
                          const platform::NCCLCommunicator *ctxs,
                          bool is_encoded = false, int nranks = -1);
  std::string Name() const override;

 protected:
  void RunImpl() override;
  int GetKValue(const std::string &grad_name);
  bool IsEncoded();
  void RunImplEncoded();
  void SparseAllReduceFunc(
      const std::vector<std::function<void()>> &all_gather_calls,
      const std::vector<std::function<void()>> &sparse_reduce_calls);

 private:
  bool is_encoded_{false};
  int nranks_{-1};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

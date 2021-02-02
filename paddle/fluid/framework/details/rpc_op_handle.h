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
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
class OpDesc;
class Scope;
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

struct RPCOpHandle : public OpHandleBase {
  RPCOpHandle(ir::Node* node, const framework::OpDesc& op_desc,
              Scope* local_scope, const std::string& name,
              const platform::Place& place);

  std::string Name() const override;

  // Delay and buffer nccl_all_reduce together can significantly increase
  // performance. Disable this feature by returning false.
  bool IsMultiDeviceTransfer() override { return false; };

 protected:
  void RunImpl() override;

  std::vector<Scope*> GetLocalScopes() override { return {local_scope_}; }

 private:
  std::unique_ptr<OperatorBase> op_;
  Scope* local_scope_;
  const std::string name_;
  platform::Place place_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle

// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"

#include <string>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {
class MemOptVarInfo;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

class ComputationOpHandle;

ComputationOpHandle *GetUniquePendingComputationOpHandle(
    ShareTensorBufferOpHandle *share_tensor_op) {
  ComputationOpHandle *result_op = nullptr;
  for (ir::Node *out_var : share_tensor_op->Node()->outputs) {
    for (ir::Node *pending_op : out_var->outputs) {
      auto &op = pending_op->Wrapper<OpHandleBase>();
      auto *compute_op = dynamic_cast<ComputationOpHandle *>(&op);
      PADDLE_ENFORCE_NOT_NULL(
          compute_op,
          platform::errors::PreconditionNotMet(
              "The pending OpHandle should be ComputationOpHandle."));

      if (result_op == nullptr) {
        result_op = compute_op;
      } else {
        PADDLE_ENFORCE_EQ(
            result_op, compute_op,
            platform::errors::PreconditionNotMet(
                "The pending OpHandle should be the unique one."));
      }
    }
  }

  PADDLE_ENFORCE_NOT_NULL(result_op,
                          platform::errors::PreconditionNotMet(
                              "The pending OpHandle should not be NULL."));
  return result_op;
}

ShareTensorBufferOpHandle::ShareTensorBufferOpHandle(
    ir::Node *node, Scope *scope, size_t scope_idx, const std::string &op_type,
    const std::vector<const ir::MemOptVarInfo *> &in_var_infos,
    const std::vector<std::string> &out_var_names, bool share_dims_and_dtype)
    : OpHandleBase(node),
      functor_(scope, scope_idx, op_type, in_var_infos, out_var_names,
               is_variant_scope_, share_dims_and_dtype) {}

std::unordered_map<std::string, std::string>
ShareTensorBufferOpHandle::ReusedVars() const {
  return functor_.ReusedVars();
}

void ShareTensorBufferOpHandle::AddReuseVarPair(
    const ir::MemOptVarInfo *in_var_info, const std::string &out_var_name) {
  functor_.AddReuseVarPair(in_var_info, out_var_name);
}

void ShareTensorBufferOpHandle::SetShareDimsAndDtype(
    bool share_dims_and_dtype) {
  functor_.SetShareDimsAndDtype(share_dims_and_dtype);
}

void ShareTensorBufferOpHandle::InitCUDA() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  int dev_id = dev_ctxes_.begin()->first.device;
  events_[dev_id] = nullptr;
#endif
}

void ShareTensorBufferOpHandle::RunImpl() { functor_(local_exec_scopes_[0]); }

}  // namespace details
}  // namespace framework
}  // namespace paddle

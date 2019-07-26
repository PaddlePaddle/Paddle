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

#include "paddle/fluid/operators/distributed/handlers/checkpoint_handler.h"

#include <string>

namespace paddle {
namespace operators {
namespace distributed {

constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";

framework::Variable* CheckpointHandler::GetOrCreateRequestVar(
    const std::string& varname, RPCRequest* request) {
  return scope_->FindVar(varname);
}

bool CheckpointHandler::Handle(RPCRequest* request) {
  if (checkpoint_block_id_ == -1) {
    LOG(WARNING) << "when checkpoint_block_id_ = -1, there should be no RPC "
                    "invoke.";
    return false;
  }

  // TODO(tangwei12): find out why scope will be error.
  auto* lt_var = scope_->FindVar(LOOKUP_TABLE_PATH)->GetMutable<std::string>();
  lt_var->clear();
  lt_var->append(request->out_var_name_);
  VLOG(4) << "RequestCheckpointHandler update var kLookupTablePath to: "
          << request->out_var_name_;
  executor_->RunPreparedContext(checkpoint_prepared_ctx_.get(), scope_);
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

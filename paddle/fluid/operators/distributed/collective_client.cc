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

#include <condition_variable>  // NOLINT
#include <string>
#include "gflags/gflags.h"

#include "paddle/fluid/operators/distributed/collective_client.h"

DECLARE_int32(rpc_deadline);

namespace paddle {
namespace operators {
namespace distributed {
std::once_flag CollectiveClient::init_flag_;
std::unique_ptr<CollectiveClient> CollectiveClient::client_(nullptr);

bool CollectiveClient::Gather(const std::vector<RemoteVar>& remote_vars,
                              std::vector<const framework::SelectedRows*>* dst,
                              const platform::DeviceContext& ctx,
                              framework::Scope* scope, int64_t time_out) {
  for (auto r : remote_vars) {
    VLOG(50) << "begin gather from ep:" << r.String();
    scope->Var(r.var_name_)->GetMutable<framework::SelectedRows>();
    VarHandlePtr ptr = rpc_client_->AsyncGetMonomerVariable(
        r.ep_, ctx, *scope, r.var_name_, time_out);
  }

  rpc_client_->Wait();

  for (auto r : remote_vars) {
    auto select_rows =
        scope->FindVar(r.var_name_)->GetMutable<framework::SelectedRows>();
    dst->push_back(select_rows);

    VLOG(4) << "gather from ep:" << r.String()
            << ", select_rows:" << GetSelectedRowsInfo(*select_rows);

    rpc_client_->AsyncGetMonomerBarrier(r.ep_, r.var_name_);
  }

  rpc_client_->Wait();
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

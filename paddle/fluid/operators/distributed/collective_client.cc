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

bool CollectiveClient::Gather(const std::vector<std::string>& eps,
                              const platform::DeviceContext& ctx,
                              const framework::Scope& scope,
                              const std::string& var_name,
                              std::vector<const framework::SelectedRows*>* dst,
                              int64_t time_out) {
  for (auto ep : eps) {
    VLOG(50) << "begin gather from ep:" << ep;
    VarHandlePtr ptr =
        rpc_client_->AsyncGetMonomerVariable(ep, ctx, scope, var_name);
    PADDLE_ENFORCE(ptr->Wait());

    auto select_rows =
        scope.FindVar(var_name)->GetMutable<framework::SelectedRows>();
    dst->push_back(select_rows);
    VLOG(40) << "gather from ep:" << ep
             << ", select_rows:" << select_rows->Info();

    rpc_client_->AsyncGetMonomerBarrier(ep, var_name);
    PADDLE_ENFORCE(ptr->Wait());

    VLOG(40) << "AsyncGetMonomerBarrier from ep:" << ep;
  }

  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

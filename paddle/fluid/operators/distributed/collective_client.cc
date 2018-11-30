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

#include "paddle/fluid/framework/selected_rows_util.h"
#include "paddle/fluid/operators/distributed/collective_client.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

DECLARE_int32(rpc_deadline);

namespace paddle {
namespace operators {
namespace distributed {

void CollectiveClient::BroadCast(const std::vector<std::string>& endpoints,
                                 const platform::DeviceContext& dev_ctx,
                                 framework::Scope* scope,
                                 const std::string& var_name,
                                 int64_t time_out) {
  distributed::RPCClient* client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

  for (auto& ep : endpoints) {
    client->AsyncSendVar(ep, dev_ctx, *scope, var_name, time_out);
  }

  client->Wait();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

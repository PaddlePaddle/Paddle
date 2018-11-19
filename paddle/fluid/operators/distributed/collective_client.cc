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

#pragma once

#include <condition_variable>  // NOLINT
#include <string>
#include "gflags/gflags.h"

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

DECLARE_int32(rpc_deadline);

namespace paddle {
namespace operators {
namespace distributed {
std::once_flag RPCClient::init_flag_;
std::unique_ptr<RPCClient> RPCClient::rpc_client_(nullptr);

bool CollectiveClient::Gather(const std::string& eps,
                              const platform::DeviceContext& ctx,
                              const framework::Scope& scope,
                              const std::string& var_name,
                              std::vector<framework::SelectedRows*>* dst,
                              int64_t time_out) {
  for (auto ep : eps) {
    VarHandlePtr ptr = rpc_client_->AsyncGetVar(ep, ctx, scope, var_name);
    ptr->Wait();

    rpc_client_->AsyncSendFetchBarrier(ep);

    auto select_rows = scope.FindVar(var_name)->GetMutable<SelectedRows>();
    dst->push_back(select_rows);
  }
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

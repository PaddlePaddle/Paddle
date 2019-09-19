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

#include "paddle/fluid/operators/distributed/rpc_client.h"
#include "gflags/gflags.h"

// default to 3min to avoid temprary network failures.
DEFINE_int32(rpc_deadline, 180000, "deadline timeouts for rpc");
DEFINE_int32(rpc_retry_times, 3, "retry times for rpc");

namespace paddle {
namespace operators {
namespace distributed {

std::once_flag RPCClient::init_flag_;
std::unique_ptr<RPCClient> RPCClient::rpc_client_(nullptr);
int RPCClient::trainer_id_ = 0;

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

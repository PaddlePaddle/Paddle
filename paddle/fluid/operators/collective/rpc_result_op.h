// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/collective/thirdparty/json.h"
#include "paddle/fluid/platform/rpc_utils.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

// service result parsers
std::vector<float> ParseScoreServiceResponse(const std::string& response) {
  return {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
}

template <typename T>
class RpcResultOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* request_id_tensor = ctx.Input<phi::DenseTensor>("X");
    std::vector<int> request_id_tensor_vec;
    framework::TensorToVector(
        *request_id_tensor, ctx.device_context(), &request_id_tensor_vec);
    int request_id = request_id_tensor_vec[0];

    // wait for call op's event notification
    auto& rpc_store = platform::RpcRequestStore::Instance();
    auto event = rpc_store.GetEvent(request_id);

    bool ok = event->wait() == 0 && rpc_store.GetErrorCode(request_id) == 0;
    if (ok) {
      const std::string& resp = rpc_store.GetResponse(request_id);
      VLOG(3) << "Request id " << request_id << " raw response: " << resp;

      // only support score service now
      auto* out = ctx.Output<phi::DenseTensor>("Out");
      ctx.device_context().Alloc<float>(out);
      auto scores = ParseScoreServiceResponse(resp);
      framework::TensorFromVector(scores, out);
    }

    auto* succeed = ctx.Output<phi::DenseTensor>("succeed");
    ctx.device_context().Alloc<bool>(succeed);
    std::vector<bool> succeed_wrapper{ok};
    framework::TensorFromVector(succeed_wrapper, succeed);
  }
};

}  // namespace operators
}  // namespace paddle

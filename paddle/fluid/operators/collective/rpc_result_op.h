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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/rpc_utils.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

// resp parsers
std::vector<double> ParseFloatResponse(const std::string& response) {
  return {json::parse(response).get<double>()};
}

std::vector<uint8_t> ParseStrResponse(const std::string& response) {
  const std::string res = json::parse(response).dump();
  return std::vector<uint8_t>(res.begin(), res.end());
}

inline void ParseResponse(phi::DenseTensor* out,
                          const std::string& res_type,
                          const platform::DeviceContext& dev_ctx,
                          const std::string& resp) {
  if (res_type == "float") {
    auto res = ParseFloatResponse(resp);
    dev_ctx.Alloc<double>(out);
    framework::TensorFromVector(res, dev_ctx, out);
  } else if (res_type == "str") {
    auto res = ParseStrResponse(resp);
    dev_ctx.Alloc<uint8_t>(out);
    framework::TensorFromVector(res, dev_ctx, out);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("Unknown result type."));
  }
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

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    int err_code = rpc_store.GetErrorCode(request_id);
    bool ok = event->wait() == 0 && err_code == 0;
    if (ok) {
      const std::string& resp = rpc_store.GetResponse(request_id);
      VLOG(3) << "Request id " << request_id << " raw response: " << resp;

      const std::string res_type = ctx.Attr<std::string>("res_type");
      VLOG(3) << "Request id " << request_id << " result type: " << res_type;

      ParseResponse(out, res_type, ctx.device_context(), resp);
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "Request %s failed with error code %s.", request_id, err_code));
    }

    auto* succeed = ctx.Output<phi::DenseTensor>("succeed");
    ctx.device_context().Alloc<bool>(succeed);
    std::vector<bool> succeed_wrapper{ok};
    framework::TensorFromVector(succeed_wrapper, ctx.device_context(), succeed);
  }
};

}  // namespace operators
}  // namespace paddle

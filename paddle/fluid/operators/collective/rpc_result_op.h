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
#include "paddle/fluid/platform/collective_helper.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

template <typename T>
class RpcResultOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* request_id_tensor = ctx.Input<phi::DenseTensor>("X");
    int request_id = request_id_tensor->data<int>()[0];

    // wait for request event
    auto* cid = platform::RpcRequestStore::Instance().GetCallId(request_id);
    auto event = platform::RpcRequestStore::Instance().GetEvent(request_id);
    event->Wait();
    brpc::Join(*cid);
    const std::string& resp =
        platform::RpcRequestStore::Instance().GetResponse(request_id);
    VLOG(3) << "Request id " << request_id << " raw response: " << resp;

    std::string res;
    const std::string service =
        platform::RpcRequestStore::Instance().GetService(request_id);
    if (service == "test") {
      json res_payload = json::parse(resp);
      json res_payload_data =
          json::parse(res_payload["data"][0].get<std::string>());

      res = "搜索结果: ";
      for (int i = 0; i < 3; ++i) {
        res += "[";
        res += std::to_string(i);
        res += "]";
        res += res_payload_data["results"][i]["abstract"];
      }
    }
    VLOG(3) << "Request id " << request_id << " result: " << res;

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::vector<uint8_t> vec(res.begin(), res.end());
    framework::TensorFromVector(vec, out);
  }
};

}  // namespace operators
}  // namespace paddle

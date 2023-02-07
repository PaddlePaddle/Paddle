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

#include <brpc/channel.h>

#include <memory>
#include <string>

#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/collective/thirdparty/json.h"
#include "paddle/fluid/platform/rpc_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

// service payload builders
std::string BuildTestServicePayload(const std::string& query) {
  json payload = {{"data", {query}}};  // => {"data": [query]}
  return payload.dump();
}

// req & res handlers
void HandleServiceRequest(brpc::Controller* ctrl,
                          int request_id,
                          const std::string& payload) {
  ctrl->request_attachment().append(payload);
  VLOG(3) << "Request id " << request_id << " payload: " << payload;
}

void HandleServiceResponse(brpc::Controller* ctrl,
                           int request_id,
                           std::shared_ptr<bthread::CountdownEvent> event) {
  // make sure the controller will be deleted
  std::unique_ptr<brpc::Controller> ctrl_guard(ctrl);
  if (ctrl->Failed()) {
    PADDLE_THROW(
        platform::errors::Unavailable("Rpc send failed: access url error."));
  }
  const std::string res = ctrl->response_attachment().to_string();
  platform::RpcRequestStore::Instance().InsertResponse(request_id, res);
  // try to notify result op
  event->signal();
}

template <typename T>
class RpcCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto InputTensor2String = [&](const std::string& name) {
      auto* tensor = ctx.Input<phi::DenseTensor>(name);
      return std::string(reinterpret_cast<const char*>(tensor->data()));
    };

    const std::string service = InputTensor2String("service");
    const std::string url = InputTensor2String("url");
    const std::string query = InputTensor2String("X");

    const std::string payload = BuildTestServicePayload(query);
    int request_id = platform::RpcSend(
        service, url, payload, &HandleServiceRequest, &HandleServiceResponse);

    VLOG(3) << "Request id " << request_id << " service: " << service;
    VLOG(3) << "Request id " << request_id << " url: " << url;
    VLOG(3) << "Request id " << request_id << " query: " << query;

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::vector<int> request_id_wrapper{request_id};
    framework::TensorFromVector(request_id_wrapper, out);
  }
};

}  // namespace operators
}  // namespace paddle

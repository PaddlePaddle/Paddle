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
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

void ResHandler(brpc::Controller* ctrl,
                int request_id,
                std::shared_ptr<platform::Semaphore> event) {
  // make sure the controller will be deleted
  std::unique_ptr<brpc::Controller> ctrl_guard(ctrl);
  if (ctrl->Failed()) {
    PADDLE_THROW(
        phi::errors::Unavailable("Rpc call op failed: access url error."));
  }
  const std::string res = ctrl->response_attachment().to_string();
  platform::RpcRequestStore::Instance().InsertResponse(request_id, res);
  // try to notify result op
  event->Signal();
}

template <typename T>
class RpcCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* service_tensor = ctx.Input<phi::DenseTensor>("service");
    const std::string service(
        reinterpret_cast<const char*>(service_tensor->data()));
    std::string url;
    if (service == "test") {
      url = "http://10.127.2.19:8082/run/predict";
    }
    PADDLE_ENFORCE_NE(
        url,
        "",
        phi::errors::InvalidArgument("Rpc call op failed: unknown service."));

    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = 10000 /*ms*/;
    options.max_retry = 3;
    PADDLE_ENFORCE_EQ(channel.Init(url.c_str(), /*load_balancer*/ "", &options),
                      0,
                      phi::errors::Unavailable(
                          "Rpc call op failed: init brpc channel error."));

    int request_id = platform::RpcRequestStore::Instance().GetRequestId();
    platform::RpcRequestStore::Instance().InsertService(request_id, service);
    VLOG(3) << "Request id " << request_id << " service: " << service;
    VLOG(3) << "Request id " << request_id << " url: " << url;

    auto* query_tensor = ctx.Input<phi::DenseTensor>("X");
    const std::string query(
        reinterpret_cast<const char*>(query_tensor->data()));
    VLOG(3) << "Request id " << request_id << " query: " << query;

    // if req is async, controller should be on heap to avoid deleting
    auto* ctrl = new brpc::Controller();
    auto cid = ctrl->call_id();
    platform::RpcRequestStore::Instance().InsertCallId(request_id, &cid);

    ctrl->http_request().uri() = url.c_str();
    ctrl->http_request().set_method(brpc::HTTP_METHOD_POST);
    ctrl->http_request().SetHeader("Content-Type", "application/json");
    if (service == "test") {
      json req_payload = {{"data", {query}}};  // => {"data": [query]}
      ctrl->request_attachment().append(req_payload.dump());
      VLOG(3) << "Request id " << request_id << " payload: " << req_payload;
    }

    auto event = std::make_shared<platform::Semaphore>();
    platform::RpcRequestStore::Instance().InsertEvent(request_id, event);
    channel.CallMethod(nullptr,
                       ctrl,
                       nullptr,
                       nullptr,
                       brpc::NewCallback(&ResHandler, ctrl, request_id, event));

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::vector<int> request_id_wrapper{request_id};
    framework::TensorFromVector(request_id_wrapper, out);
  }
};

}  // namespace operators
}  // namespace paddle

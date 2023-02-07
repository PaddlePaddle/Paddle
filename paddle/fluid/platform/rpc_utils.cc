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

#include "paddle/fluid/platform/rpc_utils.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace platform {

int RpcSend(const std::string& service,
            const std::string& url,
            const std::string& query,
            void (*payload_builder)(brpc::Controller*, int, const std::string&),
            void (*response_handler)(brpc::Controller*,
                                     int,
                                     std::shared_ptr<bthread::CountdownEvent>),
            brpc::HttpMethod http_method,
            int timeout_ms,
            int max_retry) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = timeout_ms;
  options.max_retry = max_retry;
  PADDLE_ENFORCE_EQ(
      channel.Init(url.c_str(), /*load_balancer*/ "", &options),
      0,
      phi::errors::Unavailable("Rpc send failed: init brpc channel error."));

  auto& rpc_store = RpcRequestStore::Instance();
  int request_id = rpc_store.GetRequestId();

  auto event = std::make_shared<bthread::CountdownEvent>();
  RpcRequestStore::Instance().InsertService(request_id, service);
  RpcRequestStore::Instance().InsertEvent(request_id, event);

  // if req is async, controller should be on heap to avoid deleting
  auto* ctrl = new brpc::Controller();
  ctrl->http_request().uri() = url.c_str();
  ctrl->http_request().set_method(http_method);
  ctrl->http_request().SetHeader("Content-Type", "application/json");
  payload_builder(ctrl, request_id, query);

  channel.CallMethod(
      nullptr,
      ctrl,
      nullptr,
      nullptr,
      brpc::NewCallback(response_handler, ctrl, request_id, event));

  return request_id;
}

}  // namespace platform
}  // namespace paddle

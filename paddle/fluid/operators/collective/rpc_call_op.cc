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

#include <brpc/channel.h>

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/collective/thirdparty/json.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

void ResHandler(brpc::Controller* cntl, int request_id) {
  if (cntl->Failed()) {
    PADDLE_THROW(
        phi::errors::Unavailable("Rpc call op failed: access url error."));
  }
  const std::string res_raw = cntl->response_attachment().to_string();
  VLOG(3) << res_raw;
  platform::RequestIdMap::Instance().Insert(request_id, res_raw);
}

class RpcCallOp : public framework::OperatorBase {
 public:
  RpcCallOp(const std::string& type,
            const framework::VariableNameMap& inputs,
            const framework::VariableNameMap& outputs,
            const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    int request_id = Attr<int>("request_id");
    const std::string& url = Attr<std::string>("url");
    const std::string& service_name = Attr<std::string>("service_name");
    const std::string& query = Attr<std::string>("query");

    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = 10000 /*ms*/;
    options.max_retry = 3;
    PADDLE_ENFORCE_EQ(channel.Init(url.c_str(), /*load_balancer*/ "", &options),
                      0,
                      phi::errors::Unavailable(
                          "Rpc call op failed: init brpc channel error."));

    brpc::Controller cntl;
    brpc::CallId cid = cntl.call_id();

    cntl.http_request().uri() = url.c_str();
    cntl.http_request().set_method(brpc::HTTP_METHOD_POST);
    cntl.http_request().SetHeader("Content-Type", "application/json");

    if (service_name == "test") {
      json req_payload = {{"data", {query}}};  // => {"data": [query]}
      cntl.request_attachment().append(req_payload.dump());
    }

    channel.CallMethod(nullptr,
                       &cntl,
                       nullptr,
                       nullptr,
                       brpc::NewCallback(&ResHandler, &cntl, request_id));

    brpc::Join(cid);
  }
};

class RpcCallOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<int>("request_id", "(int default 0) Unique id for request.")
        .SetDefault(0);
    AddAttr<std::string>("url", "(string default url) Service url.")
        .SetDefault("url");
    AddAttr<std::string>("service_name",
                         "(string default service_name) Service name.")
        .SetDefault("service_name");
    AddAttr<std::string>("query", "(string default query) Query to service.")
        .SetDefault("query");
    AddComment(R"DOC(
Rpc Call Operator

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(rpc_call, ops::RpcCallOp, ops::RpcCallOpMaker);

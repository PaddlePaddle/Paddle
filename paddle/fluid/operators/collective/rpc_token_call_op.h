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

#include <fstream>
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

// payload builders
template <typename T = int64_t>
static inline std::string BuildIdsPayload(const std::vector<T>& src_ids) {
  json payload = {{"ids", src_ids}};  // => {"ids": [1, 2, 3, ...]}
  return payload.dump();
}

static inline std::string BuildStrPayload(const std::string& query) {
  json payload = {{"data", {query}}};  // => {"data": [query]}
  return payload.dump();
}

template <typename T = int64_t>
static inline std::string BuildPayload(const std::string& service,
                                       const std::vector<T>& src_ids) {
  if (service == "ids") {
    return BuildIdsPayload(src_ids);
  } else if (service == "str") {
    const std::string query =
        platform::RpcTokenizer::Instance().GetWordsFromIds(src_ids);
    return BuildStrPayload(query);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("Unknown service."));
  }
}

// req & res handlers
static inline void HandleServiceRequest(brpc::Controller* ctrl,
                                        int request_id,
                                        const std::string& payload) {
  ctrl->request_attachment().append(payload);
  VLOG(3) << "Request id " << request_id << " payload: " << payload;
}

static inline void HandleServiceResponse(
    brpc::Controller* ctrl,
    int request_id,
    std::shared_ptr<bthread::CountdownEvent> event) {
  // make sure the controller will be deleted
  std::unique_ptr<brpc::Controller> ctrl_guard(ctrl);
  auto& rpc_store = platform::RpcRequestStore::Instance();
  if (ctrl->Failed()) {
    rpc_store.InsertErrorCode(request_id, ctrl->ErrorCode());
    PADDLE_THROW(platform::errors::Unavailable(
        "Request id %s failed: access url error.", request_id));
  } else {
    const std::string res = ctrl->response_attachment().to_string();
    rpc_store.InsertErrorCode(request_id, 0);
    rpc_store.InsertResponse(request_id, res);
  }
  // try to notify result op
  event->signal();
}

template <typename T>
class RpcTokenCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // url, assume num of urls is limited
    auto url_id_tensor = ctx.Input<phi::DenseTensor>("url_id");
    std::vector<int> url_id_vec;
    framework::TensorToVector(
        *url_id_tensor, ctx.device_context(), &url_id_vec);
    auto url_id = url_id_vec[0];

    auto url_list = ctx.Attr<std::vector<std::string>>("url_list");
    const std::string url = url_list[url_id];

    // payload
    auto src_ids_tensor = ctx.Input<phi::DenseTensor>("X");
    std::vector<T> src_ids_vec;
    framework::TensorToVector(
        *src_ids_tensor, ctx.device_context(), &src_ids_vec);

    bool use_ids = ctx.Attr<bool>("use_ids");
    std::string service;
    if (use_ids) {
      service = "ids";
    } else {
      // init tokenizer
      auto vocab_path = ctx.Attr<std::string>("vocab_path");
      std::unordered_map<std::string, std::string> special;
      platform::RpcTokenizer::Instance().Init(vocab_path, special);
      service = "str";
    }
    const std::string payload = BuildPayload(service, src_ids_vec);

    int request_id =
        platform::RpcCommContext::RpcSend(url,
                                          payload,
                                          &HandleServiceRequest,
                                          &HandleServiceResponse,
                                          brpc::HttpMethod::HTTP_METHOD_POST,
                                          60 * 1000,
                                          10);
    VLOG(3) << "Request id " << request_id << " url: " << url;
    VLOG(3) << "Request id " << request_id << " payload: " << payload;

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    ctx.device_context().Alloc<int>(out);
    std::vector<int> request_id_wrapper{request_id};
    framework::TensorFromVector(request_id_wrapper, ctx.device_context(), out);
  }

 private:
};

}  // namespace operators
}  // namespace paddle

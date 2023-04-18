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

#include "nlohmann/json.hpp"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/rpc_utils.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace operators {

#define DATA_STRLIST 0
/*
{"data": ["你好"]}
*/
#define TEXT_STR 1
/*
{"text": "nihao"}
*/
using json = nlohmann::json;

// payload builders
template <typename T = int64_t>
static inline std::string BuildIdsPayload(const std::vector<T> &src_ids) {
  json payload = {{"ids", src_ids}};  // => {"ids": [1, 2, 3, ...]}
  return payload.dump();
}

static inline std::string BuildStrPayload(const std::string &query,
                                          int build_way) {
  json payload;
  switch (build_way) {
    case DATA_STRLIST:
      payload = {{"data", {query}}};  //=> {"data": [query]}
      break;
    case TEXT_STR:
      payload = {{"text", query}};  //=> {"text": query}
      break;
    default:
      break;
  }

  return payload.dump();
}

template <typename T = int64_t>
static inline std::string BuildPayload(const std::string &service,
                                       const std::vector<T> &src_ids) {
  if (service == "ids") {
    return BuildIdsPayload(src_ids);
  } else if (service == "str") {
    const std::string query =
        platform::RpcTokenizer::Instance().GetWordsFromIds(src_ids);
    return BuildStrPayload(query, TEXT_STR);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("Unknown service."));
  }
}

// req & res handlers
static inline void HandleServiceRequest(brpc::Controller *ctrl,
                                        int request_id,
                                        const std::string &payload) {
  ctrl->request_attachment().append(payload);
  VLOG(3) << "Request id " << request_id << "payload size:" << payload.size();
  VLOG(3) << "Request id " << request_id << " payload: " << payload;
}

static inline void HandleServiceResponse(
    brpc::Controller *ctrl,
    int request_id,
    std::shared_ptr<bthread::CountdownEvent> event) {
  // make sure the controller will be deleted
  std::unique_ptr<brpc::Controller> ctrl_guard(ctrl);
  auto &rpc_store = platform::RpcRequestStore::Instance();
  if (ctrl->Failed()) {
    rpc_store.InsertErrorCode(request_id, ctrl->ErrorCode());
    PADDLE_THROW(platform::errors::Unavailable(
        "Request id %s failed: access url error. error code: %d, http code: %d",
        request_id,
        ctrl->ErrorCode(),
        ctrl->http_response().status_code()));
  } else {
    const std::string res = ctrl->response_attachment().to_string();
    rpc_store.InsertErrorCode(request_id, 0);
    rpc_store.InsertResponse(request_id, res);
  }
  // try to notify result op
  event->signal();
}

static int send_sequence(const framework::ExecutionContext &ctx,
                         const std::string &service,
                         const phi::DenseTensor &src_ids_tensor,
                         const std::string &url,
                         const int &timeout = 3000,
                         const int &retry = 100) {
  std::vector<int> src_ids_vec;
  framework::TensorToVector(src_ids_tensor, ctx.device_context(), &src_ids_vec);
  const std::string payload = BuildPayload(service, src_ids_vec);
  int request_id =
      platform::RpcCommContext::RpcSend(url,
                                        payload,
                                        &HandleServiceRequest,
                                        &HandleServiceResponse,
                                        brpc::HttpMethod::HTTP_METHOD_POST,
                                        timeout,
                                        retry);
  VLOG(3) << "Request id " << request_id << " url: " << url;
  VLOG(3) << "Request id " << request_id << " payload: " << payload;
  return request_id;
}

template <typename T>
class RpcCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // url, assume num of urls is limited

    const std::string url = ctx.Attr<std::string>("url");

    // payload
    auto src_ids_tensor = ctx.Input<phi::DenseTensor>("X");
    auto x_dims = src_ids_tensor->dims();

    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2,
        platform::errors::PreconditionNotMet(
            "The input src ids' dim size must be 2. However the dim is %d",
            x_dims.size()));

    std::vector<int> request_ids(x_dims[0]);

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

    int timeout = ctx.Attr<int>("timeout");
    int retry = ctx.Attr<int>("retry");

    for (auto i = 0; i < x_dims[0]; i++) {
      request_ids[i] = send_sequence(
          ctx, service, src_ids_tensor->Slice(i, i + 1), url, timeout, retry);
    }

    auto *out = ctx.Output<phi::DenseTensor>("Out");
    out->Resize({static_cast<int64_t>(request_ids.size())});
    ctx.device_context().Alloc<int>(out);
    framework::TensorFromVector(request_ids, ctx.device_context(), out);
  }
};

}  // namespace operators
}  // namespace paddle

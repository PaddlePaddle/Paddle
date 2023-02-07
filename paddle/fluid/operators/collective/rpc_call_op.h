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

// service payload builders
std::string BuildScoreServicePayload(const std::string& query) {
  json payload = {{"data", {query}}};  // => {"data": [query]}
  return payload.dump();
}

std::string BuildServicePayload(const std::string& service,
                                const std::string& query) {
  if (service == "score") {
    return BuildScoreServicePayload(query);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("Unknown service."));
  }
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
  auto& rpc_store = platform::RpcRequestStore::Instance();
  if (ctrl->Failed()) {
    LOG(WARNING) << "Request id " << request_id << " failed: access url error.";
    rpc_store.InsertStatus(request_id, false);
  } else {
    const std::string res = ctrl->response_attachment().to_string();
    rpc_store.InsertStatus(request_id, true);
    rpc_store.InsertResponse(request_id, res);
  }
  // try to notify result op
  event->signal();
}

std::unordered_map<int, std::string> GetVocabulary(std::string path) {
  std::unordered_map<int, std::string> vocab;
  std::ifstream vocab_file(path);
  std::string word;
  int id;
  while (vocab_file >> word >> id) {
    vocab.emplace(id, word);
  }
  return vocab;
}

template <typename T>
class RpcCallOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // query
    auto src_ids_tensor = ctx.Input<phi::DenseTensor>("X");
    std::vector<int> src_ids_vec;
    framework::TensorToVector(*src_ids_tensor, &src_ids_vec);

    auto vocab_path = ctx.Attr<std::string>("vocab_path");
    auto vocab = GetVocabulary(vocab_path);
    std::string query;
    for (int src_id : src_ids_vec) {
      query += vocab[src_id];
    }

    // url
    auto url_id_tensor = ctx.Input<phi::DenseTensor>("url_id");
    std::vector<int> url_id_vec;
    framework::TensorToVector(*url_id_tensor, &url_id_vec);
    int url_id = url_id_vec[0];

    auto url_list = ctx.Attr<std::vector<std::string>>("url_list");
    const std::string url = url_list[url_id];

    // payload, only support score service now
    const std::string payload = BuildServicePayload("score", query);
    int request_id = platform::RpcSend(
        url, payload, &HandleServiceRequest, &HandleServiceResponse);

    VLOG(3) << "Request id " << request_id << " url: " << url;
    VLOG(3) << "Request id " << request_id << " query: " << query;

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    std::vector<int> request_id_wrapper{request_id};
    framework::TensorFromVector(request_id_wrapper, out);
  }
};

}  // namespace operators
}  // namespace paddle

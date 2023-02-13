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

inline std::string BuildPayload(const std::vector<int>& src_ids) {
  json payload = {{"ids", src_ids}};  // => {"ids": [1, 2, 3, ...]}
  return payload.dump();
}

// req & res handlers
inline void HandleServiceRequest(brpc::Controller* ctrl,
                                 int request_id,
                                 const std::string& payload) {
  ctrl->request_attachment().append(payload);
  VLOG(3) << "Request id " << request_id << " payload: " << payload;
}

inline void HandleServiceResponse(
    brpc::Controller* ctrl,
    int request_id,
    std::shared_ptr<bthread::CountdownEvent> event) {
  // make sure the controller will be deleted
  std::unique_ptr<brpc::Controller> ctrl_guard(ctrl);
  auto& rpc_store = platform::RpcRequestStore::Instance();
  if (ctrl->Failed()) {
    VLOG(3) << "Request id " << request_id << " failed: access url error.";
    rpc_store.InsertErrorCode(request_id, ctrl->ErrorCode());
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
    // url
    auto url_id_tensor = ctx.Input<phi::DenseTensor>("url_id");
    std::vector<int> url_id_vec;
    framework::TensorToVector(
        *url_id_tensor, ctx.device_context(), &url_id_vec);
    int url_id = url_id_vec[0];

    auto url_list = ctx.Attr<std::vector<std::string>>("url_list");
    const std::string url = url_list[url_id];

    // init tokenizer
    auto vocab_path = ctx.Attr<std::string>("vocab_path");
    std::unordered_map<std::string, std::string> special;
    platform::RpcTokenizer::Instance().Init(vocab_path, special);

    // payload
    auto src_ids_tensor = ctx.Input<phi::DenseTensor>("X");
    std::vector<int> src_ids_vec;
    framework::TensorToVector(
        *src_ids_tensor, ctx.device_context(), &src_ids_vec);
    const std::string payload = BuildPayload(src_ids_vec);

    int request_id = platform::RpcCommContext::RpcSend(
        url, payload, &HandleServiceRequest, &HandleServiceResponse);
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

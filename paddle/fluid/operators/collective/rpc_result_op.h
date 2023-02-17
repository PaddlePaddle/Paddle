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

#define PARSE_DIRECT_FLOAT 0
/*
1.23
*/
#define PARSE_RESULT_FLOAT 1
/*
{"result": ["1.23"]}
*/

static inline std::vector<float> ParseFloatResponse(const std::string& response,
                                                    int parse_way) {
  auto obj = json::parse(response);
  switch (parse_way) {
    case PARSE_RESULT_FLOAT: {
      auto res = obj["result"][0].get<std::string>();
      return {std::stof(res, nullptr)};
    }
    case PARSE_DIRECT_FLOAT:
      return {obj.get<float>()};
    default:
      break;
  }
  return {static_cast<float>(0)};
}

static inline std::vector<uint8_t> ParseStrResponse(
    const std::string& response) {
  const std::string res = json::parse(response).dump();
  return std::vector<uint8_t>(res.begin(), res.end());
}

static std::vector<uint8_t> get_str_response(const int& request_id) {
  // wait for call op's event notification
  auto& rpc_store = platform::RpcRequestStore::Instance();
  auto event = rpc_store.GetEvent(request_id);
  int err_code = rpc_store.GetErrorCode(request_id);
  bool ok = event->wait() == 0 && err_code == 0;
  if (ok) {
    const std::string& resp = rpc_store.GetResponse(request_id);
    VLOG(3) << "Request id " << request_id << " raw response: " << resp;
    VLOG(3) << "Request id " << request_id;

    // auto out_ = const_cast<phi::DenseTensor&>(out);
    auto out_vector = ParseStrResponse(resp);
    return out_vector;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Request %s failed with error code %s.", request_id, err_code));
  }
}

static std::vector<float> get_float_response(const int& request_id) {
  // wait for call op's event notification
  auto& rpc_store = platform::RpcRequestStore::Instance();
  auto event = rpc_store.GetEvent(request_id);
  int err_code = rpc_store.GetErrorCode(request_id);
  bool ok = event->wait() == 0 && err_code == 0;
  if (ok) {
    const std::string& resp = rpc_store.GetResponse(request_id);
    VLOG(3) << "Request id " << request_id << " raw response: " << resp;
    VLOG(3) << "Request id " << request_id;

    // auto out_ = const_cast<phi::DenseTensor&>(out);
    auto out_vector = ParseFloatResponse(resp, PARSE_RESULT_FLOAT);
    return out_vector;
  } else {
    PADDLE_THROW(platform::errors::Unavailable(
        "Request %s failed with error code %s.", request_id, err_code));
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

    auto* out = ctx.Output<phi::DenseTensor>("Out");
    const std::string res_type = ctx.Attr<std::string>("res_type");

    VLOG(3) << "out dims: " << out->dims().to_str()
            << "numel: " << out->numel();
    if (res_type == "str") {
      ctx.device_context().Alloc<uint8_t>(out);
    } else if (res_type == "float") {
      ctx.device_context().Alloc<float>(out);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unknown result type. error type: %s", res_type.c_str()));
    }
    VLOG(3) << "out dims: " << out->dims().to_str();
    std::vector<std::vector<uint8_t>> uint8_vec;
    std::vector<std::vector<float>> float_vec;
    int64_t max_size = -1;

    for (auto i = 0; i < request_id_tensor->dims()[0]; i++) {
      if (res_type == "float") {
        auto vec = get_float_response(request_id_tensor_vec[i]);
        max_size = std::max(max_size, static_cast<int64_t>(vec.size()));
        float_vec.emplace_back(vec);
      } else if (res_type == "str") {
        auto vec = get_str_response(request_id_tensor_vec[i]);
        uint8_vec.emplace_back(vec);
        max_size = std::max(max_size, static_cast<int64_t>(vec.size()));
        PADDLE_ENFORCE_LE(
            max_size,
            100 * 1024 * 1024,
            platform::errors::Unavailable("to many string data, exceed 100MB"));
      }
    }

    out->Resize({request_id_tensor->dims()[0], max_size});

    if (res_type == "str") {
      ctx.device_context().Alloc<uint8_t>(out);
      for (size_t i = 0; i < uint8_vec.size(); i++) {
        phi::DenseTensor out_ = out->Slice(i, i + 1);
        for (int k = uint8_vec[i].size(); k < max_size; k++) {
          uint8_vec[i].emplace_back(static_cast<uint8_t>(0));
        }
        framework::TensorFromVector(uint8_vec[i], ctx.device_context(), &out_);
      }
    } else if (res_type == "float") {
      ctx.device_context().Alloc<float>(out);
      for (size_t i = 0; i < float_vec.size(); i++) {
        phi::DenseTensor out_ = out->Slice(i, i + 1);
        framework::TensorFromVector(float_vec[i], ctx.device_context(), &out_);
      }
    }

    auto* succeed = ctx.Output<phi::DenseTensor>("succeed");
    ctx.device_context().Alloc<bool>(succeed);
    std::vector<bool> succeed_wrapper{true};
    framework::TensorFromVector(succeed_wrapper, ctx.device_context(), succeed);
  }
};

}  // namespace operators
}  // namespace paddle

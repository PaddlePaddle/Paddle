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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/collective/thirdparty/json.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/phi/common/pstring.h"

namespace paddle {
namespace operators {

using json = nlohmann::json;

template <typename T>
class RpcResultOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int request_id = ctx.Attr<int>("request_id");
    const std::string& service_name = ctx.Attr<std::string>("service_name");
    const std::string& result =
        platform::RequestIdMap::Instance().GetRequestResult(request_id);

    if (service_name == "test") {
      json res_payload = json::parse(result);
      json res_payload_data =
          json::parse(res_payload["data"][0].get<std::string>());

      std::string res = "搜索结果: ";
      for (int i = 0; i < 3; ++i) {
        res += "[";
        res += std::to_string(i);
        res += "]";
        res += res_payload_data["results"][i]["abstract"];
      }

      VLOG(3) << res;

      // auto dtype = experimental::DataType::FLOAT32;
      // phi::DenseTensor data(dtype);
      // data.Resize({1});
      // data.mutable_data(ctx.GetPlace(), dtype);
      // data.data<float>()[0] = 1.0;

      // auto* out = ctx.Output<phi::DenseTensor>("Out");
      // // ctx.device_context().Alloc<T>(out);
      // out->mutable_data(ctx.GetPlace(), dtype);

      // framework::TensorCopySync(data, platform::CPUPlace(), out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

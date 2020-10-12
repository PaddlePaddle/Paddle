/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/uniform_random_op.h"
#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/xpu_header.h"

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class XPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    framework::Tensor* tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
    } else if (out_var->IsType<framework::SelectedRows>()) {
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      tensor = out_var->GetMutable<framework::SelectedRows>()->mutable_value();
      tensor->Resize(framework::make_ddim(shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    T* data = tensor->mutable_data<T>(ctx.GetPlace());

    int64_t size = tensor->numel();
    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    auto engine = framework::GetCPURandomEngine(seed);

    T* data_host = reinterpret_cast<T*>(std::malloc(size * sizeof(T)));
    for (int64_t i = 0; i < size; ++i) {
      data_host[i] = dist(*engine);
    }

    if (std::is_same<T, float>::value &&
        std::is_same<DeviceContext, platform::XPUDeviceContext>::value) {
      platform::XPUPlace place =
          BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace());
      memory::Copy(place, data, platform::CPUPlace(), data_host,
                   size * sizeof(T));
    } else {
      PADDLE_THROW(platform::errors::Unavailable(
          "Unsupported place! Only support XPU device."));
    }

    std::free(data_host);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(uniform_random,
                       paddle::operators::XPUUniformRandomKernel<
                           paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU

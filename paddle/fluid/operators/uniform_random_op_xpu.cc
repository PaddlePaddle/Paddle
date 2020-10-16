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
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

template <typename T>
class XPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    framework::Tensor *tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
    } else if (out_var->IsType<framework::SelectedRows>()) {
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      auto *selected_rows = out_var->GetMutable<framework::SelectedRows>();
      tensor = selected_rows->mutable_value();
      tensor->Resize(framework::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be "
          "LoDTensor, "
          "SelectedRows. But got unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    T *data = tensor->mutable_data<T>(ctx.GetPlace());

    int64_t size = tensor->numel();
    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    // TODO(pangyoki): implement GetXPURandomEngine to set different seeds on
    // corresponding XPU device.
    auto engine = framework::GetCPURandomEngine(seed);

    std::unique_ptr<T[]> data_cpu(new T[size]);
    for (int64_t i = 0; i < size; ++i) {
      data_cpu[i] = dist(*engine);
    }

    memory::Copy(BOOST_GET_CONST(platform::XPUPlace, ctx.GetPlace()), data,
                 platform::CPUPlace(), reinterpret_cast<void *>(data_cpu.get()),
                 size * sizeof(T));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_XPU_KERNEL(uniform_random,
                       paddle::operators::XPUUniformRandomKernel<float>);

#endif  // PADDLE_WITH_XPU

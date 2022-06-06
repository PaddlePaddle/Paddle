// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ClipMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    auto min = static_cast<T>(ctx.Attr<float>("min"));
    auto max = static_cast<T>(ctx.Attr<float>("max"));

    if (ctx.HasInput("Min")) {
      Tensor min_cpu;
      auto* min_tensor = ctx.Input<Tensor>("Min");
      auto* min_data = min_tensor->data<T>();
      if (platform::is_mlu_place(min_tensor->place())) {
        paddle::framework::TensorCopySync(*min_tensor, platform::CPUPlace(),
                                          &min_cpu);
        min_data = min_cpu.data<T>();
      }
      min = min_data[0];
    }

    if (ctx.HasInput("Max")) {
      Tensor max_cpu;
      auto* max_tensor = ctx.Input<Tensor>("Max");
      auto* max_data = max_tensor->data<T>();
      if (platform::is_mlu_place(max_tensor->place())) {
        paddle::framework::TensorCopySync(*max_tensor, platform::CPUPlace(),
                                          &max_cpu);
        max_data = max_cpu.data<T>();
      }
      max = max_data[0];
    }
    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Clip(ctx, x_desc.get(), GetBasePtr(x),
                  static_cast<const void*>(&min),
                  static_cast<const void*>(&max), GetBasePtr(out));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(clip, ops::ClipMLUKernel<float>,
                       ops::ClipMLUKernel<int32_t>,
                       ops::ClipMLUKernel<plat::float16>);

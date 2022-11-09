/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"

namespace paddle {
namespace operators {

template <typename T>
class ClipMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    auto min = static_cast<T>(ctx.Attr<float>("min"));
    auto max = static_cast<T>(ctx.Attr<float>("max"));

    if (ctx.HasInput("Min")) {
      Tensor min_cpu;
      auto* min_tensor = ctx.Input<phi::DenseTensor>("Min");
      auto* min_data = min_tensor->data<T>();
      if (platform::is_mlu_place(min_tensor->place())) {
        paddle::framework::TensorCopySync(
            *min_tensor, platform::CPUPlace(), &min_cpu);
        min_data = min_cpu.data<T>();
      }
      min = min_data[0];
    }

    if (ctx.HasInput("Max")) {
      Tensor max_cpu;
      auto* max_tensor = ctx.Input<phi::DenseTensor>("Max");
      auto* max_data = max_tensor->data<T>();
      if (platform::is_mlu_place(max_tensor->place())) {
        paddle::framework::TensorCopySync(
            *max_tensor, platform::CPUPlace(), &max_cpu);
        max_data = max_cpu.data<T>();
      }
      max = max_data[0];
    }
    out->mutable_data<T>(ctx.GetPlace());

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Clip(ctx,
                  x_desc.get(),
                  GetBasePtr(x),
                  static_cast<const void*>(&min),
                  static_cast<const void*>(&max),
                  GetBasePtr(out));
  }
};

template <typename T>
class ClipGradMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    auto* min_tensor =
        ctx.HasInput("Min") ? ctx.Input<phi::DenseTensor>("Min") : nullptr;
    auto* max_tensor =
        ctx.HasInput("Max") ? ctx.Input<phi::DenseTensor>("Max") : nullptr;

    auto min_val = ctx.Attr<float>("min");
    if (min_tensor) {
      Tensor min_data;
      framework::TensorCopy(
          *min_tensor,
          platform::CPUPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &min_data);
      ctx.template device_context<paddle::platform::MLUDeviceContext>().Wait();
      min_val = static_cast<float>(min_data.data<T>()[0]);
    }
    auto max_val = ctx.Attr<float>("max");
    if (max_tensor) {
      Tensor max_data;
      framework::TensorCopy(
          *max_tensor,
          platform::CPUPlace(),
          ctx.template device_context<platform::DeviceContext>(),
          &max_data);
      ctx.template device_context<paddle::platform::MLUDeviceContext>().Wait();
      max_val = static_cast<float>(max_data.data<T>()[0]);
    }

    MLUCnnlTensorDesc x_desc(*x);
    MLUCnnlTensorDesc dx_desc(*dx);
    MLUCnnlTensorDesc dout_desc(*dout);

    MLUCnnl::HardtanhBackward(ctx,
                              x_desc.get(),
                              GetBasePtr(x),
                              dout_desc.get(),
                              GetBasePtr(dout),
                              max_val,
                              min_val,
                              dx_desc.get(),
                              GetBasePtr(dx));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_MLU_KERNEL(clip,
                       ops::ClipMLUKernel<float>,
                       ops::ClipMLUKernel<plat::float16>);

REGISTER_OP_MLU_KERNEL(clip_grad,
                       ops::ClipGradMLUKernel<float>,
                       ops::ClipGradMLUKernel<plat::float16>);

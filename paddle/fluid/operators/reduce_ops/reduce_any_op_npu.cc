/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
class ReduceAnyNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    bool keep_dim = ctx.Attr<bool>("keep_dim");

    out->mutable_data<T>(ctx.GetPlace());

    std::vector<Tensor> inputs = {*x};
    if (ctx.HasInput("dim")) {
        auto dims = ctx.Attr<std::vector<int>>("dim");
        int dims_size = dims.size();
        Tensor dims_tensor;
        paddle::framework::TensorFromVector<int>(dims, &dims_tensor);
        dims_tensor.Resize(paddle::framework::make_ddim({1, dims_size}));
        dims_tensor.mutable_data<int>(ctx.GetPlace());
        inputs.push_back(dims_tensor);
    }
    
    auto runner = NpuOpRunner("ReduceAny", inputs, {*out}, {{"keep_dims", keep_dim}});
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(reduce_any, ops::ReduceAnyNPUKernel<bool>);
#endif
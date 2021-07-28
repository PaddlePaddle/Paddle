/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the Licnse. */

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/reduce_ops/reduce_min_max_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename DeviceContext, typename T>
class ReduceMaxNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    int in_dtype = ctx.Attr<int>("in_dtype");
    int out_dtype = ctx.Attr<int>("out_dtype");

    PADDLE_ENFORCE_EQ(
        in_dtype, -1,
        platform::errors::InvalidArgument(
            "attr in_dtype must be default %d, but got %d", -1, in_dtype));
    PADDLE_ENFORCE_EQ(
        out_dtype, -1,
        platform::errors::InvalidArgument(
            "attr out_dtype must be default %d, but got %d", -1, out_dtype));

    out->mutable_data<T>(ctx.GetPlace());

    framework::NPUAttributeMap attr_input = {{"axes", dims},
                                             {"keep_dims", keep_dim}};

    if (reduce_all) {
      std::vector<int> dim_vec;
      for (int i = 0; i < x->dims().size(); i++) {
        dim_vec.push_back(i);
      }

      attr_input = {{"axes", dim_vec}, {"keep_dims", keep_dim}};
    }

    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    const auto& runner = NpuOpRunner("ReduceMaxD", {*x}, {*out}, attr_input);
    runner.Run(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_NPU_KERNEL(
    reduce_max, ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, float>,
    ops::ReduceMaxNPUKernel<plat::NPUDeviceContext, plat::float16>);

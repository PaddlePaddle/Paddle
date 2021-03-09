/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class LookupTableV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");      // int tensor
    auto *output_t = ctx.Output<framework::LoDTensor>("Out");  // float tensor
    auto *table_t = ctx.Input<framework::LoDTensor>("W");
    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(
        table_var->IsType<framework::LoDTensor>(), true,
        platform::errors::InvalidArgument("npu only accept LoDTensor"));
    output_t->mutable_data<T>(ctx.GetPlace());
    framework::NPUAttributeMap attr_input = {{"validate_indices", false}};

    auto runner =
        NpuOpRunner("Gather", {*table_t, *ids_t}, {*output_t}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};

template <typename T>
class LookupTableV2GradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *ids_t = ctx.Input<framework::LoDTensor>("Ids");
    auto *output_grad_t =
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
    auto *table_t = ctx.Input<framework::LoDTensor>("W");
    auto *table_grad_t =
        ctx.Output<framework::LoDTensor>(framework::GradVarName("W"));
    /*
    auto *table_var = ctx.InputVar("W");
    PADDLE_ENFORCE_EQ(table_var->IsType<framework::LoDTensor>(), true,
    platform::errors::InvalidArgument("npu only accept LoDTensor"));
    */
    framework::NPUAttributeMap attr_input = {{"use_locking", true}};
    std::vector<T> vec;
    std::vector<int64_t> vec_int;

    TensorToVector(*table_t, ctx.device_context(), &vec);
    /*
    for (auto& v : vec){
        std::cout <<"table_t"<< v<<std::endl;
    }
    TensorToVector(*output_grad_t, ctx.device_context(), &vec);
    for (auto& v : vec){
        std::cout <<"output_grad_t"<< v<<std::endl;
    }
    TensorToVector(*ids_t, ctx.device_context(), &vec_int);
    for(auto & v : vec_int){
        std::cout <<"ids_t "<< v<<std::endl;
    }
    TensorToVector(*table_grad_t, ctx.device_context(), &vec);
    for (auto& v : vec){
        std::cout <<"table_grad_t"<< v<<std::endl;
    }
    // TensorToStream(std::cout, *table_t, ctx.device_context());
    */

    auto runner = NpuOpRunner("ScatterAdd", {*table_t, *ids_t, *output_grad_t},
                              {*table_grad_t}, attr_input);
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();
    runner.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    lookup_table_v2,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, double>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext,
                                paddle::platform::float16>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, int>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::LookupTableV2NPUKernel<paddle::platform::NPUDeviceContext, int64_t>);

REGISTER_OP_NPU_KERNEL(lookup_table_v2_grad,
                       ops::LookupTableV2GradNPUKernel<float>,
                       ops::LookupTableV2GradNPUKernel<double>,
                       ops::LookupTableV2GradNPUKernel<int>,
                       ops::LookupTableV2GradNPUKernel<uint8_t>,
                       ops::LookupTableV2GradNPUKernel<int64_t>);

#endif

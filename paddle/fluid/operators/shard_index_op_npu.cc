// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
template <typename T>
class ShardIndexNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(4) << "start kernel";
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int index_num = context.Attr<int>("index_num");
    int nshards = context.Attr<int>("nshards");
    int shard_id = context.Attr<int>("shard_id");
    int ignore_value = context.Attr<int>("ignore_value");

    PADDLE_ENFORCE_GT(
        index_num, 0,
        platform::errors::InvalidArgument(
            "The value 'index_num' for Op(shard_index) must be greater than 0, "
            "but the value given is %d.",
            index_num));
    PADDLE_ENFORCE_GT(nshards, 0,
                      platform::errors::InvalidArgument(
                          "The value 'nshard' for Op(shard_index) must be "
                          "greater than 0, but the value given is %d.",
                          nshards));
    PADDLE_ENFORCE_GE(
        shard_id, 0,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be greater or "
            "equal to 0, but the value given is %d.",
            shard_id));
    PADDLE_ENFORCE_LT(
        shard_id, nshards,
        platform::errors::InvalidArgument(
            "The value 'shard_id' for Op(shard_index) must be less than "
            "nshards (%d), but the value given is %d.",
            nshards, shard_id));

    int shard_size = (index_num + nshards - 1) / nshards;

    auto place = context.GetPlace();
    out->Resize(in->dims());
    out->set_lod(in->lod());
    out->mutable_data<T>(place);

    Tensor tmp(in->type());
    tmp.mutable_data<T>(framework::DDim({1}), place);
    FillNpuTensorWithConstant(&tmp, shard_size);

    Tensor condition(experimental::DataType::BOOL);
    condition.mutable_data<bool>(in->dims(), place);

    Tensor tmp2(in->type());
    tmp2.mutable_data<T>(in->dims(), place);

    Tensor tmp3(in->type());
    tmp3.mutable_data<T>(in->dims(), place);

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner runner;
    runner.AddInputs({*in, tmp});
    runner.AddOutputs({tmp2});
    runner.SetType("Mod");
    runner.Run(stream);

    NpuOpRunner runner1;
    runner1.AddInputs({*in, tmp});
    runner1.AddOutputs({tmp3});
    runner1.SetType("FloorDiv");
    runner1.Run(stream);

    FillNpuTensorWithConstant(&tmp, shard_id);
    NpuOpRunner runner2;
    runner2.AddInputs({tmp3, tmp});
    runner2.AddOutputs({condition});
    runner2.SetType("Equal");
    runner2.Run(stream);

    Tensor tmp4(in->type());
    tmp4.mutable_data<T>(in->dims(), place);
    FillNpuTensorWithConstant(&tmp4, ignore_value);
    tmp4.Resize(in->dims());

    NpuOpRunner runner3;
    runner3.AddInputs({condition, tmp2, tmp4});
    runner3.AddOutputs({*out});
    runner3.SetType("Select");
    runner3.Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(shard_index, ops::ShardIndexNPUKernel<int>,
                       ops::ShardIndexNPUKernel<int64_t>);

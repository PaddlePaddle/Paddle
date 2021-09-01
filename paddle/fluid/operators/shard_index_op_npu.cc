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

#include "paddle/fluid/operators/shard_index_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

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

    Tensor shard_size_tensor(in->type());
    shard_size_tensor.mutable_data<T>(framework::DDim({1}), place);
    FillNpuTensorWithConstant(&shard_size_tensor, static_cast<T>(shard_size));

    Tensor id_matched(framework::proto::VarType::BOOL);
    id_matched.mutable_data<bool>(in->dims(), place);

    Tensor sharded_index(in->type());
    sharded_index.mutable_data<T>(in->dims(), place);

    Tensor sharding_id(in->type());
    sharding_id.mutable_data<T>(in->dims(), place);

    Tensor shard_id_tensor(in->type());
    shard_id_tensor.mutable_data<T>(framework::DDim({1}), place);
    FillNpuTensorWithConstant(&shard_id_tensor, static_cast<T>(shard_id));

    Tensor ignore_tensor(in->type());
    ignore_tensor.mutable_data<T>(in->dims(), place);
    FillNpuTensorWithConstant(&ignore_tensor, static_cast<T>(ignore_value));
    ignore_tensor.Resize(in->dims());

    auto stream =
        context.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    NpuOpRunner("Mod", {*in, shard_size_tensor}, {sharded_index}, {})
        .Run(stream);
    NpuOpRunner("FloorDiv", {*in, shard_size_tensor}, {sharding_id}, {})
        .Run(stream);
    NpuOpRunner("Equal", {sharding_id, shard_id_tensor}, {id_matched}, {})
        .Run(stream);
    NpuOpRunner("Select", {id_matched, sharded_index, ignore_tensor}, {*out},
                {})
        .Run(stream);
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_NPU_KERNEL(shard_index, ops::ShardIndexNPUKernel<int>,
                       ops::ShardIndexNPUKernel<int64_t>);

//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class ShardIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class ShardIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, LoDTensor<int|int64>) Input variable. Each value "
             "of X is an index.");
    AddOutput(
        "Out",
        "(Tensor, Tensor<int|int64>) Output tensor with same shape as X. "
        "The tensor consists of sharding representations of values in X.");
    AddAttr<int>("index_num",
                 "A positive integer to specify the range of the input X.");

    AddAttr<int>("nshards",
                 "A positive integer to specify the number of shards.");
    AddAttr<int>("shard_id", "The current shard id");
    AddAttr<int>("ignore_value", "An integer value out of sharded range")
        .SetDefault(-1);
    AddComment(R"DOC(
This layer creates the sharded index for input. This layers is used in
model- and data- parallel mixed training generally, in which the index
data (usually the label) should be recaculated in each trainer according
to

.. math::

    assert index_num % nshards == 0

    shard_size = index_num / nshards

    y = x % shard_size if x / shard_size == shard_id else ignore_value

We take the distributed one-hot representation to show what this layer is
used for. The distributed one-hot representation is separated into multiple
shards, and each shard is filling zeros except the one with the index
inside. In order to create these sharded representation in each trainer,
the original index should be recalculated (i.e. sharded) before.

Examples:

    X is a Tensor of integer values:
      X.shape = [4, 1]
      X.data = [[1], [6], [12], [19]]

    suppose index_num = 20 and nshards = 2, then we get shard_size = 10

    if shard_id == 0, we get the Out:
      Out.shape = [4, 1]
      Out.data = [[1], [6], [-1], [-1]]

    if shard_id == 1, we get the Out:
      Out.shape = [4, 1]
      Out.data = [[-1], [-1], [2], [9]]

    the default `ignore_value` -1 is used in this example.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(shard_index,
                            ShardIndexInferShapeFunctor,
                            PD_INFER_META(phi::ShardIndexInferMeta));
REGISTER_OPERATOR(
    shard_index,
    ops::ShardIndexOp,
    ops::ShardIndexOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ShardIndexInferShapeFunctor);

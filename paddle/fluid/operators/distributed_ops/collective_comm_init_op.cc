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

#include <nccl.h>
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

class CollectiveCommInitOp : public framework::OperatorBase {
 public:
  CollectiveCommInitOp(const std::string& type,
      const framework::VariableNameMap& inputs,
      const framework::VariableNameMap& outputs,
      const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE(is_gpu_place(place),
                   "CollectiveCommInitOp can run on gpu place only.");

    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(var);
    ncclUniqueId* nccl_id = var->GetMutable<ncclUniqueId>();

    int nranks = Attr<int>("nranks");
    int rank_id = Attr<int>("rank");
    int gid = Attr<int>("group");

    platform::NCCLContextPool::Instance().CreateCommGroup(
        nccl_id, nranks, gid)->InitRank(place, rank_id);
  }
};

class CollectiveCommInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
NCCLContextInit operator

Initialize nccl context within this trainer
)DOC");
    AddAttr<int>("nranks", "(int) The number of ranks of distributed trainers");
    AddAttr<int>("rank",
        "(int) The rank of the trainer in distributed training.");
    AddAttr<int>("group", "(int) user specified group id").SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(collective_comm_init,
                  ops::CollectiveCommInitOp,
                  ops::CollectiveCommInitOpMaker);

/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include <nccl.h>
#endif
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

class CCommInitOp : public framework::OperatorBase {
 public:
  CCommInitOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE(is_gpu_place(place),
                   "CCommInitOp can run on gpu place only.");

    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(var);
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    ncclUniqueId* nccl_id = var->GetMutable<ncclUniqueId>();

    int nranks = Attr<int>("nranks");
    int rank_id = Attr<int>("rank");

    platform::NCCLReference::Instance().InitNCCLContext(nccl_id, nranks,
                                                        rank_id, place);
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
};

class CCommInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
CCommInit operator

Initialize collective communicatoin context within this trainer
)DOC");
    AddAttr<int>("nranks", "(int) The number of ranks of distributed trainers");
    AddAttr<int>("rank",
                 "(int) The rank of the trainer in distributed training.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init, ops::CCommInitOp, ops::CCommInitOpMaker);

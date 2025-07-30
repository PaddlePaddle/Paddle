/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#if defined(PADDLE_WITH_NCCL)
#include <nccl.h>
#endif
#if defined(PADDLE_WITH_RCCL)
#include <rccl.h>
#endif
#include <cstdint>

#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
// #include "paddle/fluid/operators/distributed/distributed.h"
// #include "paddle/fluid/operators/distributed/request_handler_impl.h"
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#include "paddle/phi/core/platform/collective_helper.h"
#endif

namespace paddle::operators {

class CCommInitMultiTrainerInferShape : public framework::InferShapeBase {
 public:
  ~CCommInitMultiTrainerInferShape() override = default;
  void operator()(framework::InferShapeContext* ctx) const override{};
};

class CCommInitMultiTrainerOp : public framework::OperatorBase {
 public:
  CCommInitMultiTrainerOp(const std::string& type,
                          const framework::VariableNameMap& inputs,
                          const framework::VariableNameMap& outputs,
                          const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& place) const override {
    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        var, common::errors::InvalidArgument("Input X must be provided."));
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    ncclUniqueId* nccl_id = var->GetMutable<ncclUniqueId>();

    int ntrainers = Attr<int>("ntrainers");
    int train_id = Attr<int>("trainer_id");
    int rid = Attr<int>("ring_id");

    std::vector<int> devices = Attr<std::vector<int>>("devices");

    if (devices.empty()) {
      devices = platform::GetSelectedDevices();
    }
    platform::NCCLCommContext::Instance().CreateNCCLCommMultiTrainer(
        devices, nccl_id, ntrainers, train_id, rid);
#else
    PADDLE_THROW(
        common::errors::Unimplemented("PaddlePaddle should compile with GPU."));
#endif
  }
};

class CCommInitMultiTrainerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Raw variable contains a NCCL UniqueId instances.");
    AddComment(R"DOC(
CCommInitMultiTrainer operator

Initialize collective communication context within this trainer
)DOC");
    AddAttr<int>("ntrainers",
                 "(int) The number of trainers of distributed trainers");
    AddAttr<int>("trainer_id",
                 "(int) The id of the trainer in distributed training.");
    AddAttr<std::vector<int>>("devices",
                              "(std::vector<int>) which devices does the nccl "
                              "comm initialized on in each trainer")
        .SetDefault({});
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init_multitrainer,
                  ops::CCommInitMultiTrainerOp,
                  ops::CCommInitMultiTrainerInferShape,
                  ops::CCommInitMultiTrainerOpMaker);

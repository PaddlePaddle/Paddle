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
#if defined(PADDLE_WITH_NCCL)
#include <nccl.h>
#endif
#if defined(PADDLE_WITH_RCCL)
#include <rccl.h>
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
#include "xpu/bkcl.h"
#endif
#include <string>

#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_CUSTOM_DEVICE)
#include "paddle/phi/core/platform/collective_helper.h"
#endif

#include "paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.h"
#include "paddle/phi/core/distributed/comm_context_manager.h"
#include "paddle/phi/core/distributed/store/store_utils.h"
#include "paddle/phi/core/distributed/store/tcp_store.h"

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::operators {

class CCommInitOp : public framework::OperatorBase {
 public:
  CCommInitOp(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& place) const override {
    if (place.GetType() == phi::AllocationType::CUSTOM) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
      auto var = scope.FindVar(Input("X"));
      PADDLE_ENFORCE_NOT_NULL(
          var, common::errors::InvalidArgument("Input con not be empty."));

      int nranks = Attr<int>("nranks");
      int rid = Attr<int>("ring_id");

      int device_id = place.device;
      if (Attr<int>("device_id") >= 0) {
        device_id = Attr<int>("device_id");
      }
      int rank_id = Attr<int>("rank");

      VLOG(3) << "#### use new comm lab ####";
      auto store = phi::distributed::CreateOrGetGlobalTCPStore();
      if (!phi::distributed::CommContextManager::GetInstance().Has(
              std::to_string(rid))) {
        phi::distributed::CommContextManager::CreateXCCLCommContext(
            store,
            std::to_string(rid),
            phi::CustomPlace(place.GetDeviceType(), device_id),
            rank_id,
            nranks,
            "c_comm_init_op");
      }
      return;

#else
      PADDLE_THROW(common::errors::PreconditionNotMet(
          "PaddlePaddle should compile with custom device."));
#endif
    } else {
      PADDLE_ENFORCE_EQ(place.GetType() == phi::AllocationType::GPU ||
                            place.GetType() == phi::AllocationType::XPU,
                        true,
                        common::errors::PreconditionNotMet(
                            "CCommInitOp can run on gpu or xpu place only."));

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
      auto var = scope.FindVar(Input("X"));
      PADDLE_ENFORCE_NOT_NULL(
          var, common::errors::InvalidArgument("Input con not be empty."));

      int nranks = Attr<int>("nranks");
      int rid = Attr<int>("ring_id");

      int device_id =
          static_cast<int>(static_cast<unsigned char>(place.device));
      if (Attr<int>("device_id") >= 0) {
        device_id = Attr<int>("device_id");
      }
      int rank_id = Attr<int>("rank");
#endif
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      VLOG(3) << "#### use new comm lab ####";
      auto store = phi::distributed::CreateOrGetGlobalTCPStore();
      phi::distributed::CommContextManager::SetDeviceId(device_id);
      std::string endpoints = Attr<std::string>("endpoints");
      phi::distributed::CommContextManager::CreateNCCLCommContext(
          store, std::to_string(rid), rank_id, nranks, endpoints);
#elif defined(PADDLE_WITH_XPU_BKCL)
      VLOG(3) << "#### use new comm lab ####";
      auto store = phi::distributed::CreateOrGetGlobalTCPStore();
      phi::distributed::CommContextManager::SetDeviceId(device_id);
      std::string endpoints = Attr<std::string>("endpoints");
      phi::distributed::CommContextManager::CreateBKCLCommContext(
          store, std::to_string(rid), rank_id, nranks, endpoints);
#endif
    }
  }
};

class CCommInitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Raw variable contains a NCCL UniqueId instances.");
    AddComment(R"DOC(
CCommInit operator

Initialize collective communication context within this trainer
)DOC");
    AddAttr<int>("nranks", "(int) The number of ranks of distributed trainers");
    AddAttr<int>("rank",
                 "(int) The rank of the trainer in distributed training.");
    AddAttr<int>("device_id",
                 "(int) The device_id on which to initialize the communicator."
                 "Now, you only have to set this attr manually for pipeline "
                 "training. Otherwise, make it as default.")
        .SetDefault(-1);
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
    AddAttr<std::string>("endpoints",
                         "['trainer1_ip:port', 'trainer2_ip:port', ...] "
                         "list of other trainer endpoints")
        .SetDefault("");
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init, ops::CCommInitOp, ops::CCommInitOpMaker);

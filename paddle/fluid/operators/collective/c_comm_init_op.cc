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

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/collective_helper.h"
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
    PADDLE_ENFORCE_EQ(is_gpu_place(place) || is_xpu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "CCommInitOp can run on gpu or xpu place only."));

    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input con not be empty."));
    if (is_gpu_place(place)) {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
      ncclUniqueId* nccl_id = var->GetMutable<ncclUniqueId>();

      int nranks = Attr<int>("nranks");
      int rank_id = Attr<int>("rank");
      int rid = Attr<int>("ring_id");
      int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).device;
      if (Attr<int>("device_id") >= 0) {
        device_id = Attr<int>("device_id");
      }
      platform::NCCLCommContext::Instance().CreateNCCLComm(
          nccl_id, nranks, rank_id, device_id, rid);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should be compiled with GPU."));
#endif
    } else if (is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU_BKCL)
      BKCLUniqueId* bkcl_id = var->GetMutable<BKCLUniqueId>();

      int nranks = Attr<int>("nranks");
      int rank_id = Attr<int>("rank");
      int rid = Attr<int>("ring_id");
      PADDLE_ENFORCE_EQ(
          rid, 0,
          platform::errors::OutOfRange(
              "Ring id must equal 0 in multi Kunlun cards training, but got %d",
              rid));
      int device_id = BOOST_GET_CONST(platform::XPUPlace, place).device;
      if (Attr<int>("device_id") >= 0) {
        device_id = Attr<int>("device_id");
      }
      platform::BKCLCommContext::Instance().CreateBKCLComm(
          bkcl_id, nranks, rank_id, device_id, rid);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should be compiled with XPU."));
#endif
    } else {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "CCommInitOp can run on gpu or xpu place only."));
    }
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
    AddAttr<int>("device_id",
                 "(int) The deivce_id on which to initialize the communicator."
                 "Now, you only have to set this attr manually for pipeline "
                 "training. Otherwise, make it as default.")
        .SetDefault(-1);
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_init, ops::CCommInitOp, ops::CCommInitOpMaker);

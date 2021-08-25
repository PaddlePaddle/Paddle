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

#include <string>

#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

class CCommDestroyOpAscend : public framework::OperatorBase {
 public:
  CCommDestroyOpAscend(const std::string& type,
                       const framework::VariableNameMap& inputs,
                       const framework::VariableNameMap& outputs,
                       const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE_EQ(is_npu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "CCommDestroyOpAscend can run on npu place only."));

    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input con not be empty."));
#if defined(PADDLE_WITH_ASCEND_CL)
    platform::HCCLCommContext::Instance().ReleaseHCCLComms();
    VLOG(3) << "Release hccl comms.";
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

class CCommDestroyOpAscendMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCommDestroy operator

Destroy collective communicatoin context within this trainer
)DOC");
    AddAttr<int>(
        "device_id",
        "(int) The deivce_id on which to destroy the communicator."
        "Now, you only have to set this attr manually for pipeline "
        "training. Otherwise, make it as default. Default to release all!")
        .SetDefault(-1);
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_comm_destroy_hccl, ops::CCommDestroyOpAscend,
                  ops::CCommDestroyOpAscendMaker);

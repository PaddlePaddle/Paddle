/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

class CCommInitOpNPU : public framework::OperatorBase {
 public:
  CCommInitOpNPU(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    int rid = Attr<int>("ring_id");
    int nranks = Attr<int>("nranks");
    int rank_id = Attr<int>("rank");
    int device_id = BOOST_GET_CONST(platform::NPUPlace, place).device;
    if (Attr<int>("device_id") >= 0) {
      device_id = Attr<int>("device_id");
    }
    std::vector<int> rank_ids = Attr<std::vector<int>>("rank_ids");

    VLOG(3) << "begin c_comm_init on npu, parameters are: "
            << "ring id[" << rid
            << "], nranks[" << nranks
            << "], rank_id[" << rank_id
            << "], device_id[" << device_id
            << "]";

    platform::HCCLCommContext::Instance().CreateHCCLComm(
        rank_ids, rank_id, device_id, rid);
  }
};

class CCommInitOpNPUMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
CCommInit operator on NPU

Initialize collective communication context within this trainer
)DOC");
    AddAttr<int>("nranks", "(int) The number of ranks of distributed trainers");
    AddAttr<std::vector<int>>("rank_ids", "The world rank ids of the group");
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

REGISTER_OPERATOR(c_comm_init_hcom, ops::CCommInitOpNPU, ops::CCommInitOpNPUMaker);

#endif

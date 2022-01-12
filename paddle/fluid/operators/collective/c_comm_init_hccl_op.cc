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

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle
#if defined(PADDLE_WITH_ASCEND_CL)
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

class CCommInitOpAscend : public framework::OperatorBase {
 public:
  CCommInitOpAscend(const std::string& type,
                    const framework::VariableNameMap& inputs,
                    const framework::VariableNameMap& outputs,
                    const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE_EQ(platform::is_npu_place(place), true,
                      platform::errors::PreconditionNotMet(
                          "CCommInitOpAscend can run on npu place only."));

    auto var = scope.FindVar(Input("X"));
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input con not be empty."));
#if defined(PADDLE_WITH_ASCEND_CL)
    HcclRootInfo* hccl_id = var->GetMutable<HcclRootInfo>();

    int rank_ids = Attr<int>("rank_ids");
    int rank_id = Attr<int>("rank");
    int rid = Attr<int>("ring_id");
    int device_id = place.device;
    if (Attr<int>("device_id") >= 0) {
      device_id = Attr<int>("device_id");
    }
    platform::HCCLCommContext::Instance().CreateHCCLComm(
        hccl_id, rank_ids, rank_id, device_id, rid);

    //  Build comm
    float* buff;
    int32_t size = 20;
    std::vector<float> input(size, 0);
    for (int32_t idx = 0; idx < size; idx++) {
      input[idx] = 1.0;
    }
    PADDLE_ENFORCE_NPU_SUCCESS(platform::RecordedNPUMalloc(
        reinterpret_cast<void**>(&buff), size * sizeof(float), device_id));
    platform::NPUMemcpySync(reinterpret_cast<void*>(buff), input.data(),
                            size * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE,
                            size * sizeof(float));
    VLOG(3) << "Build buff data successful.";

    aclrtStream stream = nullptr;
    auto comm = paddle::platform::HCCLCommContext::Instance().Get(rid, place);
    if (rank_id == 0) {
      stream = comm->stream();
    } else {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    }
    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclBroadcast(
        buff, size, HCCL_DATA_TYPE_FP32, 0, comm->comm(), stream));
    // Synchronize stream to find hccl error in time.
    platform::NPUStreamSync(stream);
    VLOG(3) << "Build connection successful.";
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

class CCommInitOpAscendMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
CCommInit operator

Initialize collective communicatoin context within this trainer
)DOC");
    AddAttr<int>("rank_ids",
                 "(int) The number of ranks of distributed trainers");
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

REGISTER_OPERATOR(c_comm_init_hccl, ops::CCommInitOpAscend,
                  ops::CCommInitOpAscendMaker);

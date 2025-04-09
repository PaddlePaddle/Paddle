/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/gen_comm_id_helper.h"

namespace paddle::operators {

#ifdef PADDLE_WITH_CUSTOM_DEVICE
static void GenXCCLID(std::vector<phi::ccl::CCLRootId>* xccl_ids) {
  for (size_t i = 0; i < xccl_ids->size(); ++i) {
    phi::DeviceManager::CCLGetUniqueId(
        phi::DeviceManager::GetAllCustomDeviceTypes()[0], &(*xccl_ids)[i]);
  }
}

static void CopyXCCLIDToVar(const std::vector<phi::ccl::CCLRootId>& xccl_ids,
                            std::function<std::string(size_t)> func,
                            const framework::Scope& scope) {
  for (size_t i = 0; i < xccl_ids.size(); ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound("Variable with name %s is not found",
                                 var_name.c_str()));
    auto xccl_id = var->GetMutable<phi::ccl::CCLRootId>();
    *xccl_id = xccl_ids[i];
  }
}

class CGenXCCLIdOp : public framework::OperatorBase {
 public:
  CGenXCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {
    int rank = Attr<int>("rank");
    int ring_id = Attr<int>("ring_id");

    std::function<std::string(size_t)> func = [&](size_t i) -> std::string {
      return Output("Out");
    };

    std::string endpoint = Attr<std::string>("endpoint");

    std::vector<phi::ccl::CCLRootId> xccl_ids;
    xccl_ids.resize(1);
    int server_fd = platform::SocketServer::GetInstance(endpoint).socket();
    if (rank == 0) {
      GenXCCLID(&xccl_ids);
      std::vector<std::string> endpoint_list =
          Attr<std::vector<std::string>>("other_endpoints");
      platform::SendBroadCastCommID(endpoint_list, &xccl_ids, ring_id);
    } else {
      platform::RecvBroadCastCommID(server_fd, endpoint, &xccl_ids, ring_id);
    }
    CopyXCCLIDToVar(xccl_ids, func, scope);
  }
};

#else
class CGenXCCLIdOp : public framework::OperatorBase {
 public:
  CGenXCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {}
};

#endif

class CGenXCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Raw variable contains a XCCL UniqueId instances.");
    AddComment(R"DOC(
CGenXCCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string), e.g. 127.0.0.1:6175 "
                         "current listen endpoint");
    AddAttr<std::vector<std::string>>(
        "other_endpoints",
        "['trainer1_ip:port', 'trainer2_ip:port', ...] "
        "list of other trainer endpoints")
        .SetDefault({});
    AddAttr<int>("rank",
                 "(int default 0) "
                 "The rank of the trainer in distributed training.")
        .SetDefault(0);
    AddAttr<int>("ring_id", "(int default 0) user specified ring id")
        .SetDefault(0);
  }
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_xccl_id, ops::CGenXCCLIdOp, ops::CGenXCCLIdOpMaker);

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

#include "glog/logging.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/platform/dynload/hccl.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_ASCEND_CL

static void GenHCCLID(std::vector<HcclRootInfo>* hccl_ids) {
  constexpr int timeout = 2 * 60 + 10;  // 2MSL+10s
  constexpr int retry_time = 1;
  for (size_t i = 0; i < hccl_ids->size(); ++i) {
    bool failed = true;
    for (auto retry_times = 0; retry_times * retry_time < timeout;
         ++retry_times) {
      auto err = platform::dynload::HcclGetRootInfo(&(*hccl_ids)[i]);
      if (err == 0) {
        failed = false;
        break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(retry_time));
      LOG(WARNING) << "HcclGetRootInfo failed, err is: " << err << ", retry "
                   << retry_times << " times";
    }
    if (failed) {
      PADDLE_THROW(platform::errors::External("HcclGetRootInfo failed!"));
    }
  }
}

static void CopyHCCLIDToVar(const std::vector<HcclRootInfo>& hccl_ids,
                            std::function<std::string(size_t)> func,
                            const framework::Scope& scope) {
  for (size_t i = 0; i < hccl_ids.size(); ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable with name %s is not found",
                                        var_name.c_str()));
    auto hccl_id = var->GetMutable<HcclRootInfo>();
    memcpy(hccl_id, &hccl_ids[i], sizeof(HcclRootInfo));
  }
}

class CGenHCCLIdOp : public framework::OperatorBase {
 public:
  CGenHCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    int rank = Attr<int>("rank");
    int ring_id = Attr<int>("ring_id");

    std::function<std::string(size_t)> func = [&](size_t i) -> std::string {
      return Output("Out");
    };

    std::string endpoint = Attr<std::string>("endpoint");
    int server_fd = platform::SocketServer::GetInstance(endpoint).socket();

    std::vector<HcclRootInfo> hccl_ids;
    hccl_ids.resize(1);

    if (rank == 0) {
      GenHCCLID(&hccl_ids);
      std::vector<std::string> endpoint_list =
          Attr<std::vector<std::string>>("other_endpoints");
      platform::SendBroadCastCommID(endpoint_list, &hccl_ids, ring_id);
    } else {
      platform::RecvBroadCastCommID(server_fd, endpoint, &hccl_ids, ring_id);
    }

    CopyHCCLIDToVar(hccl_ids, func, scope);
  }
};

#else

class CGenHCCLIdOp : public framework::OperatorBase {
 public:
  CGenHCCLIdOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {}
};

#endif

class CGenHCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    VLOG(3) << "ele";
    AddOutput("Out", "Raw variable contains a HCCL UniqueId instaces.");
    AddComment(R"DOC(
CGenHCCLId operator

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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(c_gen_hccl_id, ops::CGenHCCLIdOp, ops::CGenHCCLIdOpMaker);

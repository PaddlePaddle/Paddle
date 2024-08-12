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

#include <ostream>
#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/gen_comm_id_helper.h"
#include "paddle/utils/string/split.h"

namespace paddle {
namespace operators {

static void GenBKCLID(std::vector<BKCLUniqueId>* bkcl_ids) {
  for (size_t i = 0; i < bkcl_ids->size(); ++i) {
    BKCLResult_t ret = bkcl_get_unique_id(&(*bkcl_ids)[i]);
    PADDLE_ENFORCE_EQ(BKCL_SUCCESS,
                      ret,
                      common::errors::PreconditionNotMet(
                          "bkcl get unique id failed [%d]", ret));
  }
}

static void CopyBKCLIDToVar(const std::vector<BKCLUniqueId>& bkcl_ids,
                            std::function<std::string(size_t)> func,
                            const framework::Scope& scope) {
  for (size_t i = 0; i < bkcl_ids.size(); ++i) {
    std::string var_name = func(i);
    auto var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::NotFound("Variable with name %s is not found",
                                 var_name.c_str()));
    auto bkcl_id = var->GetMutable<BKCLUniqueId>();
    memcpy(bkcl_id, &bkcl_ids[i], sizeof(BKCLUniqueId));
  }
}

class GenBKCLIdOp : public framework::OperatorBase {
 public:
  GenBKCLIdOp(const std::string& type,
              const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& dev_place) const override {
    std::vector<std::string> trainers =
        Attr<std::vector<std::string>>("trainers");
    int trainer_id = Attr<int>("trainer_id");
    std::string endpoint = trainers[trainer_id];

    PADDLE_ENFORCE_GE(
        trainer_id,
        0,
        common::errors::InvalidArgument("trainer_id %d is less than 0. Its "
                                        "valid range is [0, trainer_size)"));
    PADDLE_ENFORCE_LT(
        trainer_id,
        static_cast<int>(trainers.size()),
        common::errors::OutOfRange("trainer_id %d is out of range. Its valid "
                                   "range is [0, trainer_size)",
                                   trainer_id));

    int bkcl_comm_num = Attr<int>("bkcl_comm_num");
    int use_hierarchical_allreduce = Attr<bool>("use_hierarchical_allreduce");
    int inter_nranks = Attr<int>("hierarchical_allreduce_inter_nranks");
    int inter_trainer_id = -1;
    int exter_trainer_id = -1;

    if (use_hierarchical_allreduce) {
      PADDLE_ENFORCE_GT(
          trainers.size(),
          1,
          common::errors::PreconditionNotMet(
              "The number of collective trainers %llu <= 1", trainers.size()));
      PADDLE_ENFORCE_GT(
          inter_nranks,
          1,
          common::errors::PreconditionNotMet(
              "inter_nranks %d <= 1 while in hierarchical allreduce mode",
              inter_nranks));
      PADDLE_ENFORCE_EQ(
          trainers.size() % inter_nranks,
          0,
          common::errors::PreconditionNotMet(
              "The number of trainers %llu mod inter_nranks %d is not equal 0",
              trainers.size(),
              inter_nranks));

      inter_trainer_id = trainer_id % inter_nranks;

      if (trainer_id % inter_nranks == 0) {
        exter_trainer_id = trainer_id / inter_nranks;
      }
    }

    std::ostringstream ss;
    for (size_t i = 0; i < trainers.size(); i++) {
      ss << trainers[i] << ",";
    }

    VLOG(1) << "trainer_id:" << trainer_id
            << ", use_hierarchical_allreduce:" << use_hierarchical_allreduce
            << ", bkcl_comm_num:" << bkcl_comm_num
            << ", inter_nranks:" << inter_nranks
            << ", inter_trainer_id:" << inter_trainer_id
            << ", exter_trainer_id:" << exter_trainer_id
            << ", trainers:" << ss.str();

    int server_fd = -1;
    std::vector<BKCLUniqueId> bkcl_ids;
    bkcl_ids.resize(bkcl_comm_num);

    /// 1. init flat
    std::function<std::string(size_t)> func = platform::GetFlatBKCLVarName;
    // broadcast unique id
    if (trainer_id == 0) {
      GenBKCLID(&bkcl_ids);

      // server endpoints
      std::vector<std::string> flat_endpoints;
      flat_endpoints.insert(
          flat_endpoints.begin(), trainers.begin() + 1, trainers.end());
      platform::SendBroadCastCommID(flat_endpoints, &bkcl_ids);
    } else {
      server_fd = platform::CreateListenSocket(endpoint);
      platform::RecvBroadCastCommID(server_fd, endpoint, &bkcl_ids);
    }
    CopyBKCLIDToVar(bkcl_ids, func, scope);

    /*TODO(liuyuhui) Baidu Kunlun Communication Library(BKCL) don't support
    hierarchical communication
    as NVIDIA Collective Communications Library(NCCL) in multi Nvidia GPU cards,
    and will support it later.
    */
    // close socket server
    if (trainer_id != 0) {
      platform::CloseSocket(server_fd);
    }
  }
};

class GenBKCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("BKCLID", "Raw variable contains a BKCL UniqueId instances.");
    AddComment(R"DOC(
GenBKCLId operator

For trainer 0: generate a new UniqueId and send it to all the other trainers.
For trainer 1~n: start a gRPC server to get the UniqueId, once got, stop the server.
)DOC");
    AddAttr<std::vector<std::string>>(
        "trainers",
        "['trainer0_ip:port', 'trainer1_ip:port', ...] "
        "list of all trainer endpoints")
        .SetDefault({});
    AddAttr<int>("trainer_id",
                 "(int) "
                 "The index of the trainer in distributed training.");
    AddAttr<int>("bkcl_comm_num",
                 "(int default 1) "
                 "The number of bkcl communicator num.")
        .SetDefault(1);
    AddAttr<bool>("use_hierarchical_allreduce",
                  "(bool default false) "
                  "Whether to use hierarchical allreduce.")
        .SetDefault(false);
    AddAttr<int>("hierarchical_allreduce_inter_nranks",
                 "(int default 1) "
                 "Whether to use hierarchical allreduce.")
        .SetDefault(-1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_bkcl_id, ops::GenBKCLIdOp, ops::GenBKCLIdOpMaker);

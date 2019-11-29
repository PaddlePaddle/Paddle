/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <future>  // NOLINT
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed_ops/send_recv_util.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

class CheckpointNotifyOp : public framework::OperatorBase {
 public:
  CheckpointNotifyOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    std::string dir = Attr<std::string>("dir");
    std::string lookup_table_name = Attr<std::string>("lookup_table");
    int trainer_id = Attr<int>("trainer_id");

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(trainer_id);
    for (size_t i = 0; i < epmap.size(); i++) {
      auto lookup_table_save_dir =
          string::Sprintf("%s/%s_%d", dir, lookup_table_name, i);
      rpc_client->AsyncCheckpointNotify(epmap[i], lookup_table_save_dir);
      VLOG(3) << "checkpoint notify sending lookup table: " << lookup_table_name
              << " and dir:" << dir << " to " << epmap[i];
    }
    PADDLE_ENFORCE(rpc_client->Wait(), "internal error in RPCClient");
  }
};

class CheckpointNotifyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default  127.0.0.1:6164)"
                                      "Parameter Server endpoints in the order")
        .SetDefault({"127.0.0.1:6164"});
    AddAttr<std::string>(
        "dir", "(string, default '') indicate the folder checkpoint will use");
    AddAttr<std::string>("lookup_table",
                         "(string, default '') the lookup table name");
    AddAttr<int>("trainer_id", "trainer id from 0 ~ worker_num.").SetDefault(0);
    AddComment(R"DOC(
CheckpointNotify operator

This operator will send lookup table and it's checkpoint direcoty to listen_and_serve op at
the parameter server.
)DOC");
  }
};

class CheckpointNotifyOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    checkpoint_notify, ops::CheckpointNotifyOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::CheckpointNotifyOpMaker, ops::CheckpointNotifyOpShapeInference);

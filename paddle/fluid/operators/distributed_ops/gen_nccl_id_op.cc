/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <nccl.h>
#include <stdint.h>
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/platform/nccl_helper.h"

namespace paddle {
namespace operators {

class GenNCCLIdOp : public framework::OperatorBase {
 public:
  GenNCCLIdOp(const std::string& type, const framework::VariableNameMap& inputs,
              const framework::VariableNameMap& outputs,
              const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    // put nccl id in CPUPlace
    auto& dev_ctx = *pool.Get(platform::CPUPlace());
    int trainer_id = Attr<int>("trainer_id");

    std::vector<std::string> trainers =
        Attr<std::vector<std::string>>("trainers");
    PADDLE_ENFORCE(
        trainer_id >= 0 && trainer_id < static_cast<int>(trainers.size()),
        "trainer_id:%d must be in trainers.size range", trainer_id);
    std::string endpoint = trainers[trainer_id];

    framework::Scope& local_scope = scope.NewScope();

    int nccl_comm_num = Attr<int>("nccl_comm_num");
    int use_hierarchical_allreduce = Attr<bool>("use_hierarchical_allreduce");
    int inter_nranks = Attr<int>("hierarchical_allreduce_inter_nranks");

    int inter_trainer_id = -1;
    int exter_trainer_id = -1;
    if (use_hierarchical_allreduce) {
      PADDLE_ENFORCE(trainers.size() > 1, "trainers.size():%llu < 1",
                     trainers.size());
      PADDLE_ENFORCE(inter_nranks > 1, "inter_nranks:%d < 1", inter_nranks);
      PADDLE_ENFORCE((trainers.size() % inter_nranks == 0),
                     "trainers.size():%llu mod inter_nranks:%d != 0",
                     trainers.size(), inter_nranks);

      inter_trainer_id = trainer_id % inter_nranks;

      if (trainer_id % inter_nranks == 0) {
        exter_trainer_id = trainer_id / inter_nranks;
      }
    }

    if (trainer_id != 0) {
      GetIdByServer(endpoint, &local_scope, dev_ctx, nccl_comm_num,
                    use_hierarchical_allreduce, trainer_id, inter_trainer_id,
                    exter_trainer_id);
    }

    std::ostringstream ss;
    for (size_t i = 0; i < trainers.size(); i++) {
      ss << trainers[i] << ",";
    }

    VLOG(1) << "trainer_id:" << trainer_id
            << ", use_hierarchical_allreduce:" << use_hierarchical_allreduce
            << ", inter_nranks:" << inter_nranks
            << ", inter_trainer_id:" << inter_trainer_id
            << ", exter_trainer_id:" << exter_trainer_id
            << ", trainers:" << ss.str();

    // init flat
    if (trainer_id == 0) {
      std::vector<std::string> flat_endpoints;
      flat_endpoints.insert(flat_endpoints.begin(), trainers.begin() + 1,
                            trainers.end());
      // flat nccl_id
      for (int i = 0; i < nccl_comm_num; i++) {
        std::string var_name = platform::GetFlatNCCLVarName(i);
        GenerateAndSend(&local_scope, dev_ctx, var_name, flat_endpoints);
      }
    }

    if (!use_hierarchical_allreduce) {
      return;
    }

    PADDLE_ENFORCE(trainers.size() % inter_nranks == 0,
                   "enpoints.size:%llu mod inter_nranks:%d should ==0",
                   trainers.size(), inter_nranks);
    PADDLE_ENFORCE(inter_nranks > 1, "inter_nranks:%d must > 1", inter_nranks);

    // hierarchical inter ncclid
    if (inter_trainer_id == 0) {
      std::ostringstream ss;
      ss << endpoint;
      std::vector<std::string> inter_endpoints;
      for (int i = trainer_id + 1; i < trainer_id + inter_nranks &&
                                   i < static_cast<int>(trainers.size());
           i++) {
        ss << ",";
        inter_endpoints.push_back(trainers[i]);
        ss << trainers[i];
      }
      VLOG(1) << "Hierarchical inter ring endpoints:" << ss.str();
      for (int i = 0; i < nccl_comm_num; i++) {
        std::string nccl_var_name =
            platform::GetHierarchicalInterNCCLVarName(i);
        GenerateAndSend(&local_scope, dev_ctx, nccl_var_name, inter_endpoints);
      }
    }

    // hierarchical exter ncclid
    if (exter_trainer_id == 0) {
      std::ostringstream ss;
      std::vector<std::string> exter_endpoints;
      ss << endpoint;
      for (size_t i = inter_nranks; i < trainers.size(); i += inter_nranks) {
        ss << ",";
        exter_endpoints.push_back(trainers[i]);
        ss << trainers[i];
      }
      VLOG(1) << "Hierarchical exter ring endpoints:" << ss.str();
      for (int i = 0; i < nccl_comm_num; i++) {
        std::string nccl_var_name =
            platform::GetHierarchicalExterNCCLVarName(i);
        GenerateAndSend(&local_scope, dev_ctx, nccl_var_name, exter_endpoints);
      }
    }
  }

 private:
  void GenerateAndSend(framework::Scope* scope,
                       const platform::DeviceContext& dev_ctx,
                       const std::string& nccl_id_name,
                       const std::vector<std::string>& endpoint_list) const {
    auto var = scope->FindVar(nccl_id_name);
    PADDLE_ENFORCE_NOT_NULL(var, "can't find nccl_id_var_name:%s",
                            nccl_id_name);
    auto id = var->GetMutable<ncclUniqueId>();
    PADDLE_ENFORCE(platform::dynload::ncclGetUniqueId(id));

    distributed::RPCClient* client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>(0);

    for (auto& ep : endpoint_list) {
      VLOG(3) << "sending nccl_id_var:" << nccl_id_name << " to " << ep;
      client->AsyncSendVar(ep, dev_ctx, *scope, nccl_id_name);
    }
    client->Wait();
    for (auto& ep : endpoint_list) {
      client->AsyncSendBatchBarrier(ep);
    }
    client->Wait();
    VLOG(3) << "sending completed...";
  }

  void GetIdByServer(const std::string& endpoint, framework::Scope* scope,
                     const platform::DeviceContext& dev_ctx, int nccl_comm_num,
                     bool use_hierarchical_allreduce, int trainer_id,
                     int inter_trainer_id, int exter_trainer_id) const {
    // std::string endpoint = Attr<std::string>("endpoint");
    // NOTE: Can not use unique_ptr here because the default
    // deleter will call GRPC Server's base class's dtor and
    // that will cause a wired crash.
    distributed::RequestSendHandler rpc_h(distributed::DistributedMode::kSync);
    std::unique_ptr<distributed::RPCServer> rpc_service(
        new RPCSERVER_T(endpoint, 1));

    rpc_service->RegisterRPC(distributed::kRequestSend, &rpc_h);
    rpc_h.SetRPCServer(rpc_service.get());

    framework::ProgramDesc empty_program;
    framework::Executor executor(dev_ctx.GetPlace());
    rpc_h.SetScope(scope);
    rpc_h.SetDevCtx(&dev_ctx);
    rpc_h.SetProgram(&empty_program);
    rpc_h.SetExecutor(&executor);

    std::thread server_thread(
        std::bind(&distributed::RPCServer::StartServer, rpc_service.get()));

    for (int i = 0; i < nccl_comm_num; i++) {
      rpc_service->SetCond(distributed::kRequestSend);
      VLOG(3) << "trainer_id:" << trainer_id
              << " start getting nccl id from trainer 0, nccl_comm_no:" << i;
      rpc_service->WaitBarrier(distributed::kRequestSend);
      rpc_service->ResetBarrierCounter();
    }

    if (use_hierarchical_allreduce) {
      if (inter_trainer_id > 0) {
        for (int i = 0; i < nccl_comm_num; i++) {
          rpc_service->SetCond(distributed::kRequestSend);
          VLOG(3) << "trainer_id:" << trainer_id
                  << ", inter_trainer_id:" << inter_trainer_id
                  << " start getting nccl id from inter_trainer:" << i;
          rpc_service->WaitBarrier(distributed::kRequestSend);
          rpc_service->ResetBarrierCounter();
        }
      }

      if (exter_trainer_id > 0) {
        for (int i = 0; i < nccl_comm_num; i++) {
          rpc_service->SetCond(distributed::kRequestSend);
          VLOG(3)
              << "trainer_id:" << trainer_id
              << ", exter_trainer_id:" << exter_trainer_id
              << " start getting nccl id from exter_trainer 0, nccl_comm_no:"
              << i;
          rpc_service->WaitBarrier(distributed::kRequestSend);
          rpc_service->ResetBarrierCounter();
        }
      }
    }

    VLOG(3) << "traier_id:" << trainer_id
            << ", inter_trainer_id:" << inter_trainer_id
            << ", exter_trainer_id:" << exter_trainer_id
            << " got nccl id and stop server...";
    rpc_service->ShutDown();
    VLOG(3) << "rpc server stopped";
    server_thread.join();
  }
};

class GenNCCLIdOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("NCCLID", "Raw variable contains a NCCL UniqueId instaces.");
    AddComment(R"DOC(
GenNCCLId operator

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
    AddAttr<int>("nccl_comm_num",
                 "(int default 1) "
                 "The number of nccl communicator num.")
        .SetDefault(1);
    AddAttr<bool>("use_hierarchical_allreduce",
                  "(bool default false) "
                  "Wheter to use hierarchical allreduce.")
        .SetDefault(false);
    AddAttr<int>("hierarchical_allreduce_inter_nranks",
                 "(int default 1) "
                 "Wheter to use hierarchical allreduce.")
        .SetDefault(-1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gen_nccl_id, ops::GenNCCLIdOp, ops::GenNCCLIdOpMaker);

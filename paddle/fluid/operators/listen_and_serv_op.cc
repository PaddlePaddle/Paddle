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

#include <stdint.h>
#include <sys/stat.h>
#include <ostream>
#include <thread>

#include <unistd.h>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/proto_desc.h"
#include "paddle/fluid/operators/detail/grpc_server.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"
#include "paddle/fluid/operators/detail/simple_block_queue.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlock[] = "OptimizeBlock";

void RunServer(std::shared_ptr<detail::AsyncGRPCServer> service) {
  service->RunSyncUpdate();
  VLOG(4) << "RunServer thread end";
}

static void CreateTensorFromMessageType(framework::Variable *var,
                                        sendrecv::VarType var_type) {
  if (var_type == sendrecv::VarType::LOD_TENSOR) {
    var->GetMutable<framework::LoDTensor>();
  } else if (var_type == sendrecv::VarType::SELECTED_ROWS) {
    var->GetMutable<framework::SelectedRows>();
  } else {
    PADDLE_THROW(
        "VariableMessage type %d is not in "
        "[LoDTensor, SelectedRows]",
        var_type);
  }
}

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    if (!rpc_service_) {
      std::string endpoint = Attr<std::string>("endpoint");
      rpc_service_.reset(new detail::AsyncGRPCServer(endpoint));
      server_thread_.reset(new std::thread(RunServer, rpc_service_));
    }
  }

  void Stop() override {
    detail::MessageWithName term_msg;
    term_msg.first = LISTEN_TERMINATE_MESSAGE;
    rpc_service_->Push(term_msg);
    rpc_service_->ShutDown();
    server_thread_->join();
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    framework::Scope &recv_scope = scope.NewScope();

    // FIXME(Yancey1989): initialize rpc server with lazy mode.
    rpc_service_->SetScope(&recv_scope);
    rpc_service_->SetDevCtx(&dev_ctx);
    auto ins = Inputs("X");
    auto fan_in = Attr<int>("Fanin");

    auto *block = Attr<framework::BlockDesc *>(kOptimizeBlock);
    auto *program = block->Program();
    framework::Executor executor(dev_place);

    // TODO(typhoonzero): change this to a while_op for every cluster-batch.
    bool exit_flag = false;
    // Record received sparse variables, so that
    // we could reset those after execute optimize program
    std::vector<framework::Variable *> sparse_vars;
    while (!exit_flag) {
      // Get from multiple trainers, we don't care about the order in which
      // the gradients arrives, just add suffix 0~n and merge the gradient.
      rpc_service_->SetCond(0);
      size_t recv_var_cnt = 0;
      int batch_barrier = 0;
      while (batch_barrier != fan_in) {
        const detail::MessageWithName &v = rpc_service_->Get();
        auto recv_var_name = v.first;
        if (recv_var_name == LISTEN_TERMINATE_MESSAGE) {
          LOG(INFO) << "received terminate message and exit";
          exit_flag = true;
          break;
        } else if (recv_var_name == BATCH_BARRIER_MESSAGE) {
          VLOG(3) << "recv batch barrier message";
          batch_barrier++;
          continue;
        } else {
          VLOG(3) << "received grad: " << recv_var_name;
          recv_var_cnt++;
          auto *var = recv_scope.FindVar(recv_var_name);
          if (var == nullptr) {
            LOG(ERROR) << "Can not find server side var: " << recv_var_name;
            PADDLE_THROW("Can not find server side var");
          }
          detail::DeserializeFromMessage(v.second, dev_ctx, var);
          if (var->IsType<framework::SelectedRows>()) {
            sparse_vars.push_back(var);
          }
        }
      }
      if (exit_flag) {
        rpc_service_->SetCond(1);
        rpc_service_->ShutDown();
        break;
      }
      try {
        executor.Run(*program, &recv_scope, block->ID(), /*global_block*/
                     false /*create_local_scope*/, false /*create_vars*/);
      } catch (std::exception &e) {
        LOG(ERROR) << "run sub program error " << e.what();
      }
      // Reset the received sparse variables, the sum operator would not
      // sum the input sparse variables which rows is empty at the next
      // mini-batch.
      // TODO(Yancey1989): move the reset action into an operator, we couldn't
      // have any hide logic in the operator.
      for (auto &var : sparse_vars) {
        var->GetMutable<framework::SelectedRows>()->mutable_rows()->clear();
      }
      rpc_service_->SetCond(1);
      // FIXME(typhoonzero): use another condition to sync wait clients get.
      rpc_service_->WaitClientGet(fan_in);
      sparse_vars.clear();
    }  // while(true)
  }

 protected:
  std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;
};

class ListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ListenAndServOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(
ListenAndServ operator

This operator will start a RPC server which can receive variables
from send_op and send back variables to recv_op.
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<framework::BlockDesc *>(kOptimizeBlock,
                                    "BlockID to run on server side.");
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(listen_and_serv, ops::ListenAndServOp,
                  ops::ListenAndServOpMaker);

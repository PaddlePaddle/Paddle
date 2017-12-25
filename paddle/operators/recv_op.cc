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

#include "paddle/framework/data_type.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/proto_desc.h"
#include "paddle/operators/detail/send_recv_impl.h"
#include "paddle/operators/detail/simple_block_queue.h"

namespace paddle {
namespace operators {

void RunServer(Server **rpc_server,
               std::shared_ptr<detail::SendRecvServerImpl> service,
               const std::string &server_address) {
  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  std::unique_ptr<Server> server(builder.BuildAndStart());
  *rpc_server = server.get();
  LOG(INFO) << "Server listening on " << server_address << std::endl;
  server->Wait();
}

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string &type, const framework::VariableNameMap &inputs,
         const framework::VariableNameMap &outputs,
         const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {
    if (!rpc_service_) {
      rpc_service_.reset(new detail::SendRecvServerImpl());
      std::string endpoint = Attr<std::string>("endpoint");
      server_thread_.reset(
          new std::thread(RunServer, &rpc_server_, rpc_service_, endpoint));
    }
  }

  virtual ~RecvOp() {
    rpc_server_->Shutdown();
    server_thread_->join();
  }

  std::string GetGradVarNameForTrainer(const std::string &varname) const {
    if (grads_counter_.find(varname) == grads_counter_.end()) {
      grads_counter_[varname] = 0;
    }
    char ret[256];
    snprintf(ret, sizeof(ret), "%s.trainer_%d", varname.c_str(),
             grads_counter_[varname]++);
    return std::string(ret);
  }

  void Run(const framework::Scope &scope,
           const platform::Place &dev_place) const override {
    // FIXME(typhoonzero): no new scopes for every run.
    framework::Scope &recv_scope = scope.NewScope();
    rpc_service_->SetScope(&recv_scope);
    auto param_list = Attr<std::vector<std::string>>("ParamList");
    auto grad_list = Attr<std::vector<std::string>>("GradList");
    auto trainer_count = Attr<int>("Trainers");
    size_t param_count = param_list.size();
    rpc_service_->Reset();
    // TODO(typhoonzero): change this to a while_op for every cluster-batch.
    while (true) {
      // Get from multiple trainers, we don't care about order in which
      // the gradient arrives, just add suffix 0~n then average the gradient.
      for (size_t i = 0; i < param_count * trainer_count; ++i) {
        // blocking get one var from client.
        const detail::TensorWithName &v = rpc_service_->Get();
        auto grad_var_name = v.first;
        auto it = std::find(grad_list.begin(), grad_list.end(), grad_var_name);
        std::string param_var_name;
        if (it != grad_list.end()) {
          param_var_name = param_list[it - grad_list.begin()];
        } else {
          LOG(ERROR) << "grad have no paired param found!";
        }
        VLOG(3) << "recved grad: " << grad_var_name
                << " updating param: " << param_var_name;
        auto *merged_grad = recv_scope.FindVar(grad_var_name);
        if (merged_grad == nullptr) {
          // create output of merged var.
          auto merged_var = recv_scope.Var(grad_var_name);
          merged_var->GetMutable<framework::LoDTensor>();
        }

        if (trainer_count > 1) {
          grad_var_name = this->GetGradVarNameForTrainer(grad_var_name);
        }

        auto *var = recv_scope.Var(grad_var_name);
        auto *tensor = var->GetMutable<framework::LoDTensor>();
        // FIXME(typhoonzero): do not copy
        platform::DeviceContextPool &pool = platform::DeviceContextPool::Get();
        auto &dev_ctx = *pool.Borrow(place);
        framework::CopyFrom(v.second, place, dev_ctx, tensor);
      }
      rpc_service_->Reset();

      std::string program_str = Attr<std::string>("OptimizeProgram");
      framework::proto::ProgramDesc program_desc;
      program_desc.ParseFromString(program_str);
      framework::ProgramDesc program(program_desc);
      framework::Executor executor(place);
      // Run sub graph to get optimized tensor
      try {
        executor.Run(program, &recv_scope, 0, /*global_block*/
                     false /*create_local_scope*/, false /*create_vars*/);
      } catch (std::exception &e) {
        LOG(ERROR) << "run sub program error " << e.what();
      }
      rpc_service_->Done();
      grads_counter_.clear();
    }  // while(true)
  }

 protected:
  // grpc server instance to track status and gracefully shutdown.
  // borrow an pointer from server thread.
  Server *rpc_server_{nullptr};
  // grpc send/recv service implement to register.
  std::shared_ptr<detail::SendRecvServerImpl> rpc_service_;
  std::shared_ptr<std::thread> server_thread_;
  mutable std::unordered_map<std::string, int> grads_counter_;
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  RecvOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("RX", "(Tensor) Input tensor to be optimized").AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator will recv tensor from send_op
)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<std::string>("OptimizeProgram", "type string",
                         "Serialized ProgramDesc string for recv to run.");
    AddAttr<std::vector<std::string>>(
        "ParamList", "type list of string",
        "grad->param name mapping to find which param to optimize.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        "GradList", "type list of string",
        "grad->param name mapping to find which param to optimize.")
        .SetDefault({});
    AddAttr<int>("Trainers", "type int",
                 "Number of trainers in the current cluster job")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);

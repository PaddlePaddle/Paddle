/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include "paddle/framework/executor.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/proto_desc.h"
#include "paddle/operators/detail/grpc_server.h"
#include "paddle/operators/detail/sendrecvop_utils.h"
#include "paddle/operators/detail/simple_block_queue.h"
#include "paddle/string/printf.h"

#define LISTEN_TERMINATE_MESSAGE "TERMINATE@RECV"

namespace paddle {
namespace operators {

constexpr int kCondStart = 0;
constexpr int kCondRunning = 1;
constexpr int kCondDone = 2;

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
        "VraibleMessage type %d is not in "
        "[LoDTensor, SelectedRows]",
        var_type);
  }
}

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string &type, const framework::VariableNameMap &inputs,
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

  std::string GetGradVarNameForTrainer(const std::string &varname) const {
    if (grads_counter_.find(varname) == grads_counter_.end()) {
      grads_counter_[varname] = 0;
    }
    return string::Sprintf("%s.trainer_%d", varname, grads_counter_[varname]++);
  }

  void Run(const framework::Scope &scope,
           const platform::Place &dev_place) const override {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    framework::Scope &recv_scope = scope.NewScope();

    // FIXME(Yancey1989): initialize rpc server with laze mode.
    rpc_service_->SetScope(&recv_scope);
    rpc_service_->SetDevCtx(&dev_ctx);
    auto param_list = Attr<std::vector<std::string>>("ParamList");
    auto grad_list = Attr<std::vector<std::string>>("GradList");
    auto fan_in = Attr<int>("Fanin");
    size_t param_count = param_list.size();

    std::string program_str = Attr<std::string>("OptimizeProgram");
    framework::proto::ProgramDesc program_desc;
    program_desc.ParseFromString(program_str);
    framework::ProgramDesc program(program_desc);
    framework::Executor executor(dev_place);

    // rpc_service_->Reset();
    // TODO(typhoonzero): change this to a while_op for every cluster-batch.
    bool exit_flag = false;
    while (!exit_flag) {
      // Get from multiple trainers, we don't care about the order in which
      // the gradients arrives, just add suffix 0~n and merge the gradient.
      rpc_service_->SetCond(kCondStart);
      VLOG(3) << "================ start get from service ===========";
      for (size_t i = 0; i < param_count * fan_in; ++i) {
        const detail::MessageWithName &v = rpc_service_->Get();
        auto grad_var_name = v.first;
        if (grad_var_name == LISTEN_TERMINATE_MESSAGE) {
          LOG(INFO) << "received terminate message and exit";
          exit_flag = true;
          break;
        }
        auto it = std::find(grad_list.begin(), grad_list.end(), grad_var_name);
        std::string param_var_name;
        if (it != grad_list.end()) {
          param_var_name = param_list[it - grad_list.begin()];
        } else {
          LOG(ERROR) << "grad have no paired param:" << grad_var_name;
        }
        VLOG(3) << "recved grad: " << grad_var_name
                << " updating param: " << param_var_name;
        // Assume grad_var_name must appear in global scope.
        std::string grad_var_name_trainer;
        if (fan_in > 1) {
          grad_var_name = this->GetGradVarNameForTrainer(grad_var_name);
        }
        auto *var = recv_scope.FindVar(grad_var_name);
        if (var == nullptr) {
          LOG(ERROR) << "can not find server side var: " << grad_var_name;
          PADDLE_THROW("can not find server side var");
        }
        detail::DeserializeFromMessage(v.second, dev_ctx, var);
      }
      if (exit_flag) {
        break;
      }
      // rpc_service_->Reset();
      try {
        executor.Run(program, &recv_scope, 0, /*global_block*/
                     false /*create_local_scope*/, false /*create_vars*/);
      } catch (std::exception &e) {
        LOG(ERROR) << "run sub program error " << e.what();
      }
      VLOG(3) << "================ run sub program end ===========";
      rpc_service_->SetCond(kCondDone);
      rpc_service_->WaitClientGet(param_count * fan_in);
      grads_counter_.clear();
    }  // while(true)
  }

 protected:
  std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
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
    AddAttr<int>("Fanin", "type int",
                 "Number of trainers in the current cluster job")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, ops::RecvOpMaker);

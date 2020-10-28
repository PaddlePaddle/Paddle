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

#include <stdio.h>  // for removing the port file
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <thread>  // NOLINT
#include <vector>

#include "gflags/gflags.h"

#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/math/math_function.h"

#include "paddle/fluid/operators/distributed/async_sparse_param_update_recorder.h"
#include "paddle/fluid/operators/distributed/heart_beat_monitor.h"
#include "paddle/fluid/operators/distributed/large_scale_kv.h"
#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/distributed_ops/heter_listen_and_serv_op.h"

#include "paddle/fluid/platform/profiler.h"

DEFINE_int32(rpc_send_thread_num, 12, "number of threads for rpc send");
DEFINE_int32(rpc_get_thread_num, 12, "number of threads for rpc get");
DEFINE_int32(rpc_prefetch_thread_num, 12, "number of threads for rpc prefetch");

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<distributed::RPCServer> service) {
  service->StartServer();
  VLOG(4) << "RunServer thread end";
}

static void split(const std::string &str, char sep,
                  std::vector<std::string> *pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

HeterListenAndServOp::HeterListenAndServOp(
    const std::string &type, const framework::VariableNameMap &inputs,
    const framework::VariableNameMap &outputs,
    const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

HeterListenAndServOp::~HeterListenAndServOp() { Stop(); }

void HeterListenAndServOp::Stop() {
  rpc_service_->ShutDown();
  server_thread_->join();
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
}

void HeterListenAndServOp::SavePort() const {
  // NOTE: default write file to /tmp/paddle.selected_port
  rpc_service_->SavePort();
}

void HeterListenAndServOp::RunAsyncLoop(framework::Executor *executor,
                                        framework::ProgramDesc *program,
                                        framework::Scope *recv_scope) const {
  VLOG(2) << "RunAsyncLoop";
  auto grad_to_block_id_str =
      Attr<std::vector<std::string>>("grad_to_block_id");
  DoubleFindMap<std::string, int32_t> grad_to_block_id;

  auto append_block_maps = [](DoubleFindMap<std::string, int32_t> *out_map,
                              const std::string &grad_and_id) {
    std::vector<std::string> pieces;
    split(grad_and_id, ':', &pieces);
    VLOG(3) << "after split, key = " << pieces[0] << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2,
                      platform::errors::PreconditionNotMet(
                          "Invalid format of grad_and_id argument. "
                          "Expected \"grad:block_id\". Recieved %s",
                          grad_and_id.c_str()));
    PADDLE_ENFORCE_EQ(out_map->count(pieces[0]), 0,
                      platform::errors::AlreadyExists(
                          "The gradient name %s has already existed in out_map",
                          pieces[0].c_str()));

    int block_id = std::stoi(pieces[1]);
    (*out_map)[pieces[0]] = block_id;
  };

  for (const auto &grad_and_id : grad_to_block_id_str) {
    append_block_maps(&grad_to_block_id, grad_and_id);
  }

  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    platform::errors::PreconditionNotMet(
                        "Invalid number of blocks in server program. Expected "
                        "equal or greater than 2. Recieved %zu",
                        num_blocks));
  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  auto optimize_prepared = executor->Prepare(*program, block_list);
  // execute global block if needed, block id 1 in the program is global
  // block if it's not bind to a grad var for it's update.
  if (block_list[0] == 1 &&
      grad_to_block_id.find_value(static_cast<int32_t>(1)) ==
          grad_to_block_id.end()) {
    executor->RunPreparedContext(optimize_prepared[0].get(), recv_scope);
  }
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared_ctx, param_to_prepared_ctx;
  for (size_t i = 0; i < block_list.size(); ++i) {
    auto blkid = block_list[i];
    auto it = grad_to_block_id.find_value(blkid);
    if (it != grad_to_block_id.end()) {
      grad_to_prepared_ctx[it->first] = optimize_prepared[i];
    }
  }

  request_send_and_recv_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);

  while (true) {
    if (rpc_service_->IsExit()) {
      VLOG(4) << "get exit!rpc_processor break!";
      break;
    }

    sleep(1);
  }  // while(true)
}

static void FillRequestCtx(distributed::RequestHandler *h,
                           framework::Scope *scope,
                           platform::DeviceContext *dev_ctx,
                           framework::Executor *executor,
                           framework::ProgramDesc *program,
                           std::unordered_map<std::string, std::string>
                               *sparse_grad_name_to_param_name,
                           distributed::RPCServer *rpc_server) {
  h->SetScope(scope);
  h->SetDevCtx(dev_ctx);
  h->SetExecutor(executor);
  h->SetProgram(program);
  h->SetSparseGradToParam(sparse_grad_name_to_param_name);
  h->SetRPCServer(rpc_server);
}

void HeterListenAndServOp::CacheVarsType(
    const std::vector<std::string> &varnames,
    const framework::Scope &scope) const {
  for (const auto &varname : varnames) {
    auto var = scope.FindVar(varname);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::PreconditionNotMet(
                 "Received var is not initialized in the received scope."));
    if (var->IsType<framework::SelectedRows>()) {
      sparse_vars_.push_back(varname);
    } else if (var->IsType<framework::LoDTensor>() ||
               var->IsType<framework::Tensor>()) {
      dense_vars_.push_back(varname);
    } else {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "The type of received var should be in [SelectedRows, LoDTensor, "
          "Tensor]."));
    }
  }
}

void HeterListenAndServOp::RunImpl(const framework::Scope &scope,
                                   const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  int distributed_mode = Attr<int>("distributed_mode");
  auto fan_in = Attr<int>("Fanin");
  auto pserver_id = Attr<int>("pserver_id");
  auto inputs = Inputs("X");

  PADDLE_ENFORCE_EQ(rpc_service_, nullptr,
                    platform::errors::PreconditionNotMet(
                        "RPC service has been created unexpectedly."));
  std::string endpoint = Attr<std::string>("endpoint");

  VLOG(4) << "pserver_id: " << pserver_id
          << ", distributed_mode:" << distributed_mode << ", fan_in:" << fan_in
          << ", end_point:" << endpoint;

  rpc_service_.reset(new RPCSERVER_T(endpoint, fan_in));
  rpc_service_->UsingOriginRpcService();
  auto rpc_exec_thread_num = Attr<int>("rpc_exec_thread_num");

  request_send_and_recv_handler_.reset(
      new distributed::RequestSendAndRecvHandler(distributed_mode));

  rpc_service_->RegisterRPC(distributed::kRequestSendAndRecv,
                            request_send_and_recv_handler_.get(),
                            rpc_exec_thread_num);

  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE_GE(optimize_blocks.size(), 1,
                    platform::errors::PreconditionNotMet(
                        "optimize blocks is less than 1. Optimize blocks "
                        "should be 1 at least on the pserver side."));
  auto *program = optimize_blocks[0]->Program();

  framework::Executor executor(dev_place);

  // parse attr of kSparseGradToParam  sparse_grad_name -> param_name
  std::unordered_map<std::string, std::string> sparse_grad_name_to_param_name;
  auto sparse_grad_name_to_param_name_str =
      Attr<std::vector<std::string>>(kSparseGradToParam);
  for (const auto &sparse_grad_name_and_param_name :
       sparse_grad_name_to_param_name_str) {
    std::vector<std::string> pieces;
    split(sparse_grad_name_and_param_name, ':', &pieces);
    PADDLE_ENFORCE_EQ(
        pieces.size(), 2,
        platform::errors::PreconditionNotMet(
            "Invalid format of sparse_grad_name_and_param_name argument. "
            "Expected \"xxx:xxx\". Recieved %s",
            sparse_grad_name_and_param_name.c_str()));
    VLOG(3) << "after split, sparse_grad_name = " << pieces[0]
            << ", param_name = " << pieces[1];
    sparse_grad_name_to_param_name[pieces[0]] = pieces[1];
  }

  auto f = std::bind(FillRequestCtx, std::placeholders::_1, &recv_scope,
                     &dev_ctx, &executor, program,
                     &sparse_grad_name_to_param_name, rpc_service_.get());

  f(request_send_and_recv_handler_.get());

  // register SIGINT(from ctrl+C) and SIGTERM(from kill) signal handlers
  signal(SIGINT, SignalHandler::StopAndExit);
  signal(SIGTERM, SignalHandler::StopAndExit);

  VLOG(2) << "RunAsyncLoop";
  auto grad_to_block_id_str =
      Attr<std::vector<std::string>>("grad_to_block_id");

  if (grad_to_block_id_str.size() == 0) {
    VLOG(0) << "there are no gradients on this parameter server";
  } else {
    std::vector<std::string> pieces;
    split(grad_to_block_id_str[0], ':', &pieces);
    distributed::HeartBeatMonitor::Init(fan_in, pserver_id == 0, pieces[0]);
  }

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();
  // Write to a file of server selected port for python use.
  SavePort();
  RunAsyncLoop(&executor, program, &recv_scope);
}

class HeterListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(
        R"DOC(" + "HeterListenAndServ operator" + "\n" + "This operator" +
" will start a RPC server which can receive variables from send_op and send" +
"back variables to recv_op.)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<int>("pserver_id",
                 "(int, default -1), the parameter server index id")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>(
        "grad_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<int>("distributed_mode",
                 "indicate distriubte training mode, 0 is sync, 1 is "
                 "fully-async, 2 is half-async, 3 is geo")
        .SetDefault(0);
    AddAttr<std::vector<framework::BlockDesc *>>(
        kOptimizeBlocks, "Optimize blocks to run on server side.")
        .SetDefault({});
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
    AddAttr<int>("rpc_exec_thread_num", "pserver send thread num.")
        .SetDefault(1);
  }
};

void SignalHandler::StopAndExit(int signal_num) {
  // Do not use VLOG here for the device for printing maybe already released.
  // exit will release interal allocated resoureces.
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
  exit(0);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(heter_listen_and_serv, ops::HeterListenAndServOp,
                  ops::HeterListenAndServOpMaker);

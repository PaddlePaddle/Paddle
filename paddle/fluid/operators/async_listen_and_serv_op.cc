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

#include <fstream>
#include <ostream>
#include <thread>  // NOLINT
#include <vector>

#include "paddle/fluid/operators/async_listen_and_serv_op.h"

#include "paddle/utils/StringUtil.h"

namespace paddle {
namespace operators {

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

void RunServer(std::shared_ptr<detail::SyncGRPCServer> service) {
  service->RunAsyncUpdate();
  VLOG(4) << "RunServer thread end";
}

static void AsyncExecuteBlock(framework::Executor *executor,
                              framework::ExecutorPrepareContext *prepared,
                              framework::Scope *scope) {
  framework::Async([&executor, &prepared, &scope]() {
    try {
      executor->RunPreparedContext(prepared, scope, false, false);
    } catch (std::exception &e) {
      LOG(ERROR) << "run sub program error " << e.what();
    }
  });
}

AsyncListenAndServOp::AsyncListenAndServOp(
    const std::string &type, const framework::VariableNameMap &inputs,
    const framework::VariableNameMap &outputs,
    const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

int AsyncListenAndServOp::GetSelectedPort() const {
  return rpc_service_->GetSelectedPort();
}

void AsyncListenAndServOp::Stop() {
  rpc_service_->Push(LISTEN_TERMINATE_MESSAGE);
  server_thread_->join();
}

void AsyncListenAndServOp::RunImpl(const framework::Scope &scope,
                                   const platform::Place &dev_place) const {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  if (!rpc_service_) {
    std::string endpoint = Attr<std::string>("endpoint");
    rpc_service_.reset(new detail::SyncGRPCServer(endpoint));
  }

  // grad name to block id
  std::unordered_map<std::string, int32_t> grad_to_id;
  std::unordered_map<int32_t, std::string> id_to_grad;

  auto grad_map_str = Attr<std::vector<std::string>>("grad_to_id");
  for (auto &grad_and_id : grad_map_str) {
    std::vector<std::string> pieces;
    split(grad_and_id, ' ', &pieces);
    PADDLE_ENFORCE_EQ(pieces.size(), 2);
    PADDLE_ENFORCE_EQ(grad_to_id.count(pieces[0]), 0);
    int block_id = std::stoi(pieces[1]);
    grad_to_id[pieces[0]] = block_id;
    id_to_grad[block_id] = pieces[0];
  }

  auto *optimize_block = Attr<framework::BlockDesc *>(kOptimizeBlock);
  auto *prefetch_block = Attr<framework::BlockDesc *>(kPrefetchBlock);
  auto *program = optimize_block->Program();
  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  framework::Executor executor(dev_place);
  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    if (blkid != static_cast<size_t>(prefetch_block->ID())) {
      block_list.push_back(blkid);
    }
  }
  PADDLE_ENFORCE_EQ(grad_map_str.size(), block_list.size(),
                    "grad num should be equal to optimize block num");
  auto optimize_prepared = executor.Prepare(*program, block_list);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared;
  for (size_t i = 0; i < block_list.size(); ++i) {
    grad_to_prepared[id_to_grad[block_list[i]]] = optimize_prepared[i];
  }

  rpc_service_->SetScope(&recv_scope);
  rpc_service_->SetDevCtx(&dev_ctx);

  // set proper fields for table lookup and update
  rpc_service_->SetExecutor(&executor);
  VLOG(3) << "prefetch block id is " << prefetch_block->ID();
  auto prefetch_prepared = executor.Prepare(*program, prefetch_block->ID());
  rpc_service_->SetPrefetchPreparedCtx(prefetch_prepared.get());
  prefetch_prepared.release();
  rpc_service_->SetProgram(program);

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  sleep(5);
  // Write to a file of server selected port for python use.
  std::ofstream port_file;
  port_file.open("/tmp/paddle.selected_port");
  port_file << rpc_service_->GetSelectedPort();
  port_file.close();

  bool exit_flag = false;
  while (!exit_flag) {
    const detail::ReceivedMessage v = rpc_service_->Get();
    auto recv_var_name = v.first;
    if (recv_var_name == LISTEN_TERMINATE_MESSAGE) {
      LOG(INFO) << "received terminate message and exit";
      exit_flag = true;
      break;
    } else {
      VLOG(3) << "received grad: " << recv_var_name;
      auto var = v.second->GetVar();
      if (var == nullptr) {
        LOG(ERROR) << "Can not find server side var: " << recv_var_name;
        PADDLE_THROW("Can not find server side var");
      }
      AsyncExecuteBlock(&executor, grad_to_prepared[recv_var_name].get(),
                        &recv_scope);
      // TODO(qiao): explain why
      if (var->IsType<framework::SelectedRows>()) {
        var->GetMutable<framework::SelectedRows>()->mutable_rows()->clear();
      }
    }

    if (exit_flag) {
      rpc_service_->ShutDown();
      break;
    }
  }  // while(true)
}

class AsyncListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AsyncListenAndServOpMaker(OpProto *proto, OpAttrChecker *op_checker)
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
    AddAttr<std::vector<std::string>>(
        "grad_to_id(['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'])",
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<framework::BlockDesc *>(kOptimizeBlock,
                                    "BlockID to run on server side.");
    AddAttr<framework::BlockDesc *>(kPrefetchBlock,
                                    "prefetch block to run on server side.");
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(async_listen_and_serv, ops::AsyncListenAndServOp,
                  ops::AsyncListenAndServOpMaker);

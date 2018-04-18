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

namespace paddle {
namespace operators {

void RunServer(std::shared_ptr<detail::SyncGRPCServer> service) {
  service->RunAsyncUpdate();
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

static void ParallelExecuteBlocks(
    const std::vector<size_t> &parallel_blkids, framework::Executor *executor,
    const std::vector<std::shared_ptr<framework::ExecutorPrepareContext>>
        &prepared,
    framework::ProgramDesc *program, framework::Scope *scope) {
  std::vector<std::future<void>> fs;
  for (size_t idx : parallel_blkids) {
    fs.push_back(
        framework::Async([&executor, &prepared, &program, &scope, idx]() {
          int run_block = idx;  // thread local
          try {
            executor->RunPreparedContext(prepared[run_block].get(), scope,
                                         false, false);
          } catch (std::exception &e) {
            LOG(ERROR) << "run sub program error " << e.what();
          }
        }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

static void AsyncExecuteBlock(
    size_t block_id, framework::Executor *executor,
    std::shared_ptr<framework::ExecutorPrepareContext> ctx,
    framework::ProgramDesc *program, framework::Scope *scope) {}

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
  auto optimize_prepared = executor.Prepare(*program, block_list);
  // Insert placeholder for block0 which holds current op itself.
  optimize_prepared.insert(
      optimize_prepared.begin(),
      std::shared_ptr<framework::ExecutorPrepareContext>(nullptr));

  rpc_service_->SetScope(&recv_scope);
  rpc_service_->SetDevCtx(&dev_ctx);
  // TODO(qiao) set proper fields for table lookup and update
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
  // Record received sparse variables, so that
  // we could reset those after execute optimize program
  std::vector<framework::Variable *> sparse_vars;
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
      if (var->IsType<framework::SelectedRows>()) {
        sparse_vars.push_back(var);
      }
      AsyncExecuteBlock();
    }

    if (exit_flag) {
      rpc_service_->ShutDown();
      break;
    }

    // NOTE: if is_gpu_place, CUDA kernels are launched by multiple threads
    // and this will still work.

    // The optimize blocks which have the same parent ID would run parallel
    // TODO(Yancey1989): need to use ParallelExecutor for future
    int32_t last_parent_blkid = program->Block(1).Parent();

    VLOG(2) << "run all blocks spent " << detail::GetTimestamp() - ts << "(ms)";

    // Reset the received sparse variables, the sum operator would not
    // sum the input sparse variables which rows is empty at the next
    // mini-batch.
    // TODO(Yancey1989): move the reset action into an operator, we couldn't
    // have any hide logic in the operator.
    for (auto &var : sparse_vars) {
      var->GetMutable<framework::SelectedRows>()->mutable_rows()->clear();
    }
    // FIXME(typhoonzero): use another condition to sync wait clients get.
    sparse_vars.clear();
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

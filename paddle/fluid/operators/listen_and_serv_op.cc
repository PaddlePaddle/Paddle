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

#include "paddle/fluid/operators/detail/macros.h"

#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include "paddle/fluid/operators/listen_and_serv_op.h"
#include "paddle/fluid/platform/profiler.h"

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
            executor->RunPreparedContext(prepared[run_block].get(), scope);
          } catch (const std::exception &e) {
            LOG(ERROR) << "run sub program error " << e.what();
          }
        }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

ListenAndServOp::ListenAndServOp(const std::string &type,
                                 const framework::VariableNameMap &inputs,
                                 const framework::VariableNameMap &outputs,
                                 const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

ListenAndServOp::~ListenAndServOp() { Stop(); }

void ListenAndServOp::Stop() {
  rpc_service_->ShutDown();
  server_thread_->join();
  auto file_path = string::Sprintf("/tmp/paddle.%d.port", ::getpid());
  remove(file_path.c_str());
}

void ListenAndServOp::SavePort() const {
  // NOTE: default write file to /tmp/paddle.selected_port
  rpc_service_->SavePort();
}

static int64_t GetTimestamp() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void ListenAndServOp::RunSyncLoop(
    framework::Executor *executor, framework::ProgramDesc *program,
    framework::Scope *recv_scope,
    const std::vector<int> &prefetch_block_id_list) const {
  size_t num_blocks = program->Size();
  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  std::vector<int> optimize_blocks_idx;
  for (auto blk : optimize_blocks) {
    optimize_blocks_idx.push_back(blk->ID());
  }
  auto optimize_prepared = executor->Prepare(*program, optimize_blocks_idx);
  // Insert placeholder for block0 which holds current op itself.
  optimize_prepared.insert(
      optimize_prepared.begin(),
      std::shared_ptr<framework::ExecutorPrepareContext>(nullptr));

  rpc_service_->ResetBarrierCounter();
  while (true) {
    // Get from multiple trainers, we don't care about the order in which
    // the gradients arrives, just add suffix 0~n and merge the gradient.
    rpc_service_->SetCond(distributed::kRequestSend);
    rpc_service_->WaitBarrier(distributed::kRequestSend);

    if (rpc_service_->IsExit()) {
      LOG(WARNING) << "get exit!rpc_processor break!";
      rpc_service_->SetCond(distributed::kRequestGet);
      break;
    }

    // NOTE: if is_gpu_place, CUDA kernels are launched by multiple threads
    // and this will still work.
    // The optimize blocks which have the same parent ID would run parallel
    // TODO(Yancey1989): need to use ParallelExecutor for future
    int32_t last_parent_blkid = optimize_blocks[0]->Parent();
    std::vector<size_t> parallel_blkids;
    parallel_blkids.push_back(optimize_blocks[0]->ID());
    double ts = GetTimestamp();
    for (size_t i = 1; i < optimize_blocks.size(); ++i) {
      // skip the first optimize block because it is already in the
      // parallel_blkids.
      int blkid = optimize_blocks[i]->ID();
      if (program->Block(blkid).Parent() != last_parent_blkid) {
        ParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared,
                              program, recv_scope);
        parallel_blkids.clear();
        last_parent_blkid = program->Block(blkid).Parent();
      }
      parallel_blkids.push_back(blkid);
    }
    ParallelExecuteBlocks(parallel_blkids, executor, optimize_prepared, program,
                          recv_scope);
    VLOG(2) << "run all blocks spent " << GetTimestamp() - ts << "(ms)";

    rpc_service_->SetCond(distributed::kRequestGet);
    rpc_service_->WaitBarrier(distributed::kRequestGet);
    rpc_service_->ResetBarrierCounter();
    // reset received sparse vars to avoid reuse it in the next mini-batch
    dynamic_cast<distributed::RequestSendHandler *>(request_send_handler_.get())
        ->ResetSparseVarRecorder();
  }  // while(true)
}

void ListenAndServOp::RunAsyncLoop(framework::Executor *executor,
                                   framework::ProgramDesc *program,
                                   framework::Scope *recv_scope) const {
  // grad name to block id
  std::unordered_map<std::string, int32_t> grad_to_block_id;
  std::unordered_map<int32_t, std::string> id_to_grad;

  auto grad_to_block_id_str =
      Attr<std::vector<std::string>>("grad_to_block_id");
  for (const auto &grad_and_id : grad_to_block_id_str) {
    std::vector<std::string> pieces;
    split(grad_and_id, ':', &pieces);
    VLOG(3) << "after split, grad = " << pieces[0] << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2);
    PADDLE_ENFORCE_EQ(grad_to_block_id.count(pieces[0]), 0);

    int block_id = std::stoi(pieces[1]);
    grad_to_block_id[pieces[0]] = block_id;
    id_to_grad[block_id] = pieces[0];
  }
  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 2,
                    "server program should have at least 2 blocks");

  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  auto optimize_prepared = executor->Prepare(*program, block_list);
  // execute global block if needed
  if (block_list[0] == 1 && id_to_grad.count(1) == 0) {
    executor->RunPreparedContext(optimize_prepared[0].get(), recv_scope);
  }
  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      grad_to_prepared_ctx;
  for (size_t i = 0; i < block_list.size(); ++i) {
    grad_to_prepared_ctx[id_to_grad[block_list[i]]] = optimize_prepared[i];
  }

  request_send_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);
  request_get_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);
  request_prefetch_handler_->SetGradToPreparedCtx(&grad_to_prepared_ctx);

  while (true) {
    if (rpc_service_->IsExit()) {
      LOG(INFO) << "get exit!rpc_processor break!";
      break;
    }

    sleep(1);
  }  // while(true)
}

static void FillRequestCtx(
    distributed::RequestHandler *h, framework::Scope *scope,
    platform::DeviceContext *dev_ctx, framework::Executor *executor,
    framework::ProgramDesc *program,
    std::unordered_map<std::string,
                       std::shared_ptr<framework::ExecutorPrepareContext>>
        *prefetch_ctx,
    distributed::RPCServer *rpc_server) {
  h->SetScope(scope);
  h->SetDevCtx(dev_ctx);
  h->SetExecutor(executor);
  h->SetProgram(program);
  h->SetPrefetchPreparedCtx(prefetch_ctx);
  h->SetRPCServer(rpc_server);
}

void ListenAndServOp::RunImpl(const framework::Scope &scope,
                              const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::SetProfileListener();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  framework::Scope &recv_scope = scope.NewScope();

  bool sync_mode = Attr<bool>("sync_mode");
  auto fan_in = Attr<int>("Fanin");

  PADDLE_ENFORCE(!rpc_service_);
  std::string endpoint = Attr<std::string>("endpoint");

  LOG(INFO) << "sync_mode:" << sync_mode << ", fan_in:" << fan_in
            << ", end_point:" << endpoint;

  rpc_service_.reset(new RPCSERVER_T(endpoint, fan_in));

  request_send_handler_.reset(new distributed::RequestSendHandler(sync_mode));
  request_get_handler_.reset(new distributed::RequestGetHandler(sync_mode));
  request_prefetch_handler_.reset(
      new distributed::RequestPrefetchHandler(sync_mode));

  rpc_service_->RegisterRPC(distributed::kRequestSend,
                            request_send_handler_.get());
  rpc_service_->RegisterRPC(distributed::kRequestGet,
                            request_get_handler_.get());
  rpc_service_->RegisterRPC(distributed::kRequestPrefetch,
                            request_prefetch_handler_.get());

  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>(kOptimizeBlocks);
  PADDLE_ENFORCE(optimize_blocks.size() >= 1,
                 "optimize blocks should be 1 at least on the pserver side.");
  auto *program = optimize_blocks[0]->Program();
  framework::Executor executor(dev_place);

  // prepare for prefetch
  std::vector<int> prefetch_block_id_list;
  std::unordered_map<int, std::string> block_id_to_prefetch_var_name;

  auto prefetch_var_name_to_block_id_str =
      Attr<std::vector<std::string>>(kPrefetchVarNameToBlockId);
  for (const auto &prefetch_var_name_and_id :
       prefetch_var_name_to_block_id_str) {
    std::vector<std::string> pieces;
    split(prefetch_var_name_and_id, ':', &pieces);
    VLOG(3) << "after split, prefetch_var = " << pieces[0]
            << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2);

    int block_id = std::stoi(pieces[1]);
    prefetch_block_id_list.push_back(block_id);
    block_id_to_prefetch_var_name[block_id] = pieces[0];
  }

  auto prefetch_prepared = executor.Prepare(*program, prefetch_block_id_list);

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>
      prefetch_var_name_to_prepared_ctx;
  for (size_t i = 0; i < prefetch_block_id_list.size(); ++i) {
    auto block_id = prefetch_block_id_list[i];
    auto prefetch_var_name = block_id_to_prefetch_var_name[block_id];
    prefetch_var_name_to_prepared_ctx[prefetch_var_name] = prefetch_prepared[i];
  }

  auto f = std::bind(FillRequestCtx, std::placeholders::_1, &recv_scope,
                     &dev_ctx, &executor, program,
                     &prefetch_var_name_to_prepared_ctx, rpc_service_.get());

  f(request_send_handler_.get());
  f(request_get_handler_.get());
  f(request_prefetch_handler_.get());

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, rpc_service_));
  VLOG(3) << "wait server thread to become ready...";
  rpc_service_->WaitServerReady();

  // register SIGINT(from ctrl+C) and SIGTERM(from kill) signal handlers
  signal(SIGINT, SignalHandler::StopAndExit);
  signal(SIGTERM, SignalHandler::StopAndExit);

  // Write to a file of server selected port for python use.
  SavePort();
  if (sync_mode) {
    RunSyncLoop(&executor, program, &recv_scope, prefetch_block_id_list);
  } else {
    RunAsyncLoop(&executor, program, &recv_scope);
  }
}

class ListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(" + "ListenAndServ operator" + "\n" + "This operator" +
" will start a RPC server which can receive variables from send_op and send" +
"back variables to recv_op.)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string &ip) { return !ip.empty(); });
    AddAttr<std::vector<std::string>>(
        "grad_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<bool>("sync_mode", "if works at sync_mode or not").SetDefault(true);
    AddAttr<std::vector<framework::BlockDesc *>>(
        kOptimizeBlocks, "Optimize blocks to run on server side.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(kPrefetchVarNameToBlockId,
                                      "prefetch blocks to run on server side.")
        .SetDefault({});
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
  }
};

void SignalHandler::StopAndExit(int signal_num) {
  // Do not use VLOG here for the device for printing maybe already released.
  // exit will release interal allocated resoureces.
  exit(0);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(listen_and_serv, ops::ListenAndServOp,
                  ops::ListenAndServOpMaker);

/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/pscore/heter_listen_and_serv_op.h"
#include "paddle/fluid/framework/op_registry.h"

PADDLE_DEFINE_EXPORTED_int32(rpc_send_thread_num, 12,
                             "number of threads for rpc send");

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

HeterListenAndServOp::HeterListenAndServOp(
    const std::string &type, const framework::VariableNameMap &inputs,
    const framework::VariableNameMap &outputs,
    const framework::AttributeMap &attrs)
    : OperatorBase(type, inputs, outputs, attrs) {}

HeterListenAndServOp::~HeterListenAndServOp() { Stop(); }

void HeterListenAndServOp::Stop() {}

void HeterListenAndServOp::RunAsyncLoop(framework::ProgramDesc *program) const {
  VLOG(2) << "RunAsyncLoop";
  auto message_to_block_id_str =
      Attr<std::vector<std::string>>("message_to_block_id");
  DoubleFindMap<std::string, int32_t> message_to_block_id;

  auto append_block_maps = [](DoubleFindMap<std::string, int32_t> *out_map,
                              const std::string &grad_and_id) {
    std::vector<std::string> pieces;
    split(grad_and_id, ':', &pieces);
    VLOG(3) << "after split, key = " << pieces[0] << ", id=" << pieces[1];
    PADDLE_ENFORCE_EQ(pieces.size(), 2,
                      platform::errors::PreconditionNotMet(
                          "Invalid format of message_and_id argument. "
                          "Expected \"message:block_id\". Recieved %s",
                          grad_and_id.c_str()));
    PADDLE_ENFORCE_EQ(out_map->count(pieces[0]), 0,
                      platform::errors::AlreadyExists(
                          "The message name %s has already existed in out_map",
                          pieces[0].c_str()));

    int block_id = std::stoi(pieces[1]);
    (*out_map)[pieces[0]] = block_id;
  };

  for (const auto &message_and_id : message_to_block_id_str) {
    append_block_maps(&message_to_block_id, message_and_id);
  }

  size_t num_blocks = program->Size();
  PADDLE_ENFORCE_GE(num_blocks, 1,
                    platform::errors::PreconditionNotMet(
                        "Invalid number of blocks in server program. Expected "
                        "equal or greater than 1. Recieved %zu",
                        num_blocks));
  std::vector<int> block_list;
  for (size_t blkid = 1; blkid < num_blocks; ++blkid) {
    block_list.push_back(blkid);
  }
  for (size_t i = 0; i < block_list.size(); ++i) {
    auto blkid = block_list[i];
    auto it = message_to_block_id.find_value(blkid);
    heter_server_->RegisterServiceHandler(
        it->first, [&](const MultiVarMsg *request, MultiVarMsg *response,
                       brpc::Controller *cntl) -> int {
          return send_and_recv_variable_handler_->Handle(request, response,
                                                         cntl);
        });
  }

  while (true) {
    if (heter_server_->IsExit() || heter_server_->IsStop()) {
      heter_server_->Stop();
      VLOG(0) << "get exit. rpc_processor stop!";
      break;
    }
    sleep(1);
  }  // while(true)
}

void RunServer(
    std::shared_ptr<paddle::distributed::HeterServer> heter_server_ptr) {
  heter_server_ptr->StartHeterService();
}

void HeterListenAndServOp::RunImpl(const framework::Scope &scope,
                                   const platform::Place &dev_place) const {
  // Mark this as PS that it should decide profiling by listening from trainer.
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(dev_place);
  VLOG(1) << "HeterListenAndServOp::RunImpl On gpu? "
          << platform::is_gpu_place(dev_place);

  auto pserver_id = Attr<int>("pserver_id");
  auto fan_in = Attr<int>("fanin");
  auto inputs = Inputs("X");

  PADDLE_ENFORCE_EQ(heter_server_, nullptr,
                    platform::errors::PreconditionNotMet(
                        "RPC service has been created unexpectedly."));

  std::string endpoint = Attr<std::string>("endpoint");
  VLOG(4) << "pserver_id: " << pserver_id << ", end_point:" << endpoint;

  heter_server_ = distributed::HeterServer::GetInstance();
  heter_server_->SetEndPoint(endpoint);
  heter_server_->SetFanin(fan_in);

  auto optimize_blocks =
      Attr<std::vector<framework::BlockDesc *>>("optimize_blocks");
  PADDLE_ENFORCE_GE(optimize_blocks.size(), 1,
                    platform::errors::PreconditionNotMet(
                        "optimize blocks is less than 1. Optimize blocks "
                        "should be 1 at least on the pserver side."));

  auto *program = optimize_blocks[0]->Program();

  send_and_recv_variable_handler_.reset(
      new distributed::SendAndRecvVariableHandler());
  send_and_recv_variable_handler_->SetScope(&scope);
  send_and_recv_variable_handler_->SetDevCtx(&dev_ctx);
  heter_server_->SetServiceHandler(send_and_recv_variable_handler_);

  VLOG(2) << "RunAsyncLoop";

  // start the server listening after all member initialized.
  server_thread_.reset(new std::thread(RunServer, heter_server_));
  VLOG(3) << "wait server thread to become ready...";
  heter_server_->WaitServerReady();
  RunAsyncLoop(program);
  VLOG(3) << "Wait for Server_thread_ stop";
  (server_thread_.get())->join();
  VLOG(3) << "Server_thread_ stop";
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
        "message_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from message name to it's optimize block id")
        .SetDefault({});
    AddAttr<int>("distributed_mode",
                 "indicate distriubte training mode, 0 is sync, 1 is "
                 "fully-async, 2 is half-async, 3 is geo")
        .SetDefault(0);
    AddAttr<std::vector<framework::BlockDesc *>>(
        "optimize_blocks", "Optimize blocks to run on server side.")
        .SetDefault({});
    AddAttr<int>("fanin", "How many clients send to this server.")
        .SetDefault(1);
    AddAttr<int>("rpc_exec_thread_num", "pserver send thread num.")
        .SetDefault(1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(heter_listen_and_serv, ops::HeterListenAndServOp,
                  ops::HeterListenAndServOpMaker);

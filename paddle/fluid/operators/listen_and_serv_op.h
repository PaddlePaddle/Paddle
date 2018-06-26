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

#pragma once

#include <stdint.h>
#include <atomic>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlocks[] = "optimize_blocks";
constexpr char kPrefetchVarNameToBlockId[] = "prefetch_var_name_to_block_id";

void RunServer(std::shared_ptr<distributed::RPCServer> service);

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs);

  virtual ~ListenAndServOp();

  void RunSyncLoop(framework::Executor* executor,
                   framework::ProgramDesc* program,
                   framework::Scope* recv_scope,
                   const std::vector<int>& prefetch_block_id_list) const;

  void RunAsyncLoop(framework::Executor* executor,
                    framework::ProgramDesc* program,
                    framework::Scope* recv_scope) const;

  void SavePort() const;

  int GetSelectedPort() { return rpc_service_->GetSelectedPort(); }

  void Stop() override;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override;

 protected:
  mutable std::shared_ptr<distributed::RPCServer> rpc_service_;
  mutable std::shared_ptr<distributed::RequestHandler> request_send_handler_;
  mutable std::shared_ptr<distributed::RequestHandler> request_get_handler_;
  mutable std::shared_ptr<distributed::RequestHandler>
      request_prefetch_handler_;

  mutable std::shared_ptr<std::thread> server_thread_;
};

class SignalHandler {
 public:
  static void StopAndExit(int signal_num);

 private:
  DISABLE_COPY_AND_ASSIGN(SignalHandler);
};

}  // namespace operators
}  // namespace paddle

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
#include <ostream>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlock[] = "OptimizeBlock";
constexpr char kPrefetchBlock[] = "PrefetchBlock";

void RunServer(std::shared_ptr<detail::AsyncGRPCServer> service);

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs);

  int GetSelectedPort() const;

  void RunSyncLoop(framework::Executor* executor,
                   framework::ProgramDesc* program,
                   framework::Scope* recv_scope,
                   framework::BlockDesc* prefetch_block) const;

  void RunAsyncLoop(framework::Executor* executor,
                    framework::ProgramDesc* program,
                    framework::Scope* recv_scope,
                    framework::BlockDesc* prefetch_block) const;

  void Stop() override;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override;

 protected:
  mutable std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
  mutable std::shared_ptr<std::thread> server_thread_;
};

}  // namespace operators
}  // namespace paddle

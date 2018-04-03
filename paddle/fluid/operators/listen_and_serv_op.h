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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/operators/detail/grpc_server.h"

namespace paddle {
namespace operators {

constexpr char kOptimizeBlock[] = "OptimizeBlock";

void RunServer(std::shared_ptr<detail::AsyncGRPCServer> service);

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

static void ParallelExecuteBlocks(const std::vector<size_t> &parallel_blkids,
                                  framework::Executor *executor,
                                  framework::ProgramDesc *program,
                                  framework::Scope *scope) {
  std::vector<std::future<void>> fs;
  for (size_t idx : parallel_blkids) {
    fs.push_back(framework::Async([&executor, &program, &scope, idx]() {
      int run_block = idx;  // thread local
      try {
        executor->Run(*program, scope, run_block, false, false);
      } catch (std::exception &e) {
        LOG(ERROR) << "run sub program error " << e.what();
      }
    }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
}

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs);

  int GetSelectedPort();

  void Stop() override;

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override;

 protected:
  mutable std::shared_ptr<detail::AsyncGRPCServer> rpc_service_;
  mutable std::shared_ptr<std::thread> server_thread_;
};

}  // namespace operators
}  // namespace paddle

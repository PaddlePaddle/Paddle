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

#include "paddle/framework/executor.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/scope.h"

#include <boost/range/adaptor/reversed.hpp>

namespace paddle {
namespace framework {

const std::string kFeedOpType = "feed";
const std::string kFetchOpType = "fetch";

Executor::Executor(const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(), 0);
  device_contexts_.resize(places.size());
  for (size_t i = 0; i < places.size(); i++) {
    if (platform::is_cpu_place(places[i])) {
      device_contexts_[i] = new platform::CPUDeviceContext(
          boost::get<platform::CPUPlace>(places[i]));
    } else if (platform::is_gpu_place(places[i])) {
#ifdef PADDLE_WITH_CUDA
      device_contexts_[i] = new platform::CUDADeviceContext(
          boost::get<platform::GPUPlace>(places[i]));
#else
      PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
    }
  }
}

Executor::~Executor() {
  for (auto& device_context : device_contexts_) {
    delete device_context;
  }
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id) {
  // TODO(tonyyang-svail):
  //    - only runs on the first device (i.e. no interdevice communication)
  //    - will change to use multiple blocks for RNN op and Cond Op
  PADDLE_ENFORCE_GT(pdesc.blocks_size(), block_id);
  auto& block = pdesc.blocks(block_id);
  auto& device = device_contexts_[0];

  // Instantiate all the vars in the global scope
  for (auto& var : block.vars()) {
    scope->NewVar(var.name());
  }

  Scope& local_scope = scope->NewScope();

  std::vector<bool> should_run = Prune(pdesc, block_id);
  PADDLE_ENFORCE_EQ(should_run.size(), block.ops_size());
  for (size_t i = 0; i < should_run.size(); ++i) {
    if (should_run[i]) {
      for (auto& var : block.ops(i).outputs()) {
        for (auto& argu : var.arguments()) {
          if (local_scope.FindVar(argu) == nullptr) {
            local_scope.NewVar(argu);
          }
        }
      }
      LOG(INFO) << block.ops(i).type();
      if (block.ops(i).type() == "sum") {
        LOG(INFO) << "Here";
        for (auto& var : block.ops(i).inputs()) {
          for (auto& argu : var.arguments()) {
            LOG(INFO) << var.parameter() << " " << argu;
          }
        }
      }
      auto op = paddle::framework::OpRegistry::CreateOp(block.ops(i));
      LOG(INFO) << op->DebugString();
      op->Run(local_scope, *device);
    }
  }

  // TODO(tonyyang-svail):
  //  - Destroy local_scope
}

std::vector<bool> Executor::Prune(const ProgramDesc& pdesc, int block_id) {
  // TODO(tonyyang-svail):
  //    - will change to use multiple blocks for RNN op and Cond Op

  auto& block = pdesc.blocks(block_id);
  auto& ops = block.ops();

  bool expect_feed = true;
  for (auto& op_desc : ops) {
    PADDLE_ENFORCE(op_desc.type() != kFeedOpType || expect_feed,
                   "All FeedOps are at the beginning of the ProgramDesc");
    expect_feed = (op_desc.type() == kFeedOpType);
  }

  bool expect_fetch = true;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;
    PADDLE_ENFORCE(op_desc.type() != kFetchOpType || expect_fetch,
                   "All FetchOps must at the end of the ProgramDesc");
    expect_fetch = (op_desc.type() == kFetchOpType);
  }

  std::set<std::string> dependent_vars;
  std::vector<bool> should_run;
  for (auto op_iter = ops.rbegin(); op_iter != ops.rend(); ++op_iter) {
    auto& op_desc = *op_iter;

    bool found_dependent_vars = false;
    for (auto& var : op_desc.outputs()) {
      for (auto& argu : var.arguments()) {
        if (dependent_vars.count(argu) != 0) {
          found_dependent_vars = true;
        }
      }
    }

    if (op_desc.type() == kFetchOpType || found_dependent_vars) {
      // erase its output to the dependency graph
      for (auto& var : op_desc.outputs()) {
        for (auto& argu : var.arguments()) {
          dependent_vars.erase(argu);
        }
      }

      // insert its input to the dependency graph
      for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
          dependent_vars.insert(argu);
        }
      }

      LOG(INFO) << "1 " << op_desc.type();
      should_run.push_back(true);
    } else {
      LOG(INFO) << "0 " << op_desc.type();
      should_run.push_back(false);
    }
  }

  // TODO(tonyyang-svail):
  //    - check this after integration of Init
  // PADDLE_ENFORCE(dependent_vars.empty());

  // since we are traversing the ProgramDesc in reverse order
  // we reverse the should_run vector
  std::reverse(should_run.begin(), should_run.end());

  return should_run;
}

}  // namespace framework
}  // namespace paddle

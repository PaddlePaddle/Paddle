//   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_GCU

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device/gcu/executor/gcu_executor.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace operators {
using paddle::framework::Scope;
using paddle::framework::ir::Graph;
using paddle::platform::gcu::GcuExecutor;
using paddle::platform::gcu::GcuExecutorManager;
using paddle::platform::gcu::kGcuProgramKey;

class GcuRuntimeOp : public framework::OperatorBase {
 public:
  GcuRuntimeOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const {
    VLOG(10) << "=== Start to run gcu_run_time op === ";
    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    framework::RuntimeContext runtime_ctx(inputs_, outputs_, scope);
    framework::ExecutionContext ctx(*this, scope, *dev_ctx, runtime_ctx);
    auto inputs = ctx.MultiInput<phi::DenseTensor>("FeedList");
    auto outputs = ctx.MultiOutput<phi::DenseTensor>("FetchList");
    VLOG(4) << "GcuRuntime Kernel, begin to run graph";
    auto manager = GcuExecutorManager::GetInstance();
    auto attr = ctx.GetAttr(kGcuProgramKey);
    auto program_key = PADDLE_GET_CONST(std::string, attr);
    std::shared_ptr<GcuExecutor> gcu_exec = manager->Find(program_key);
    if (gcu_exec == nullptr) {
      VLOG(6) << "== target == scope ptr:" << (int64_t)(&scope)
              << ", program key:" << program_key;
      gcu_exec = std::make_shared<GcuExecutor>(&scope);
      gcu_exec->RunGcuOp(inputs, outputs, ctx, program_key);
      manager->Add(program_key, gcu_exec);
    } else {
      VLOG(6) << "== scope ptr:" << (int64_t)(&scope)
              << ", program key:" << program_key;
      gcu_exec->ResetScope(&scope);
      gcu_exec->RunGcuOp(inputs, outputs, ctx, program_key);
    }
    // auto gcu_backend = platform::gcu::GcuBackend::GetInstance();
    // gcu_backend->RunGcuOp(inputs, outputs, ctx);
  }
};

class GcuRuntimeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FeedList", "FeedList of Graph").AsDuplicable();
    AddOutput("FetchList", "FetchList of Graph").AsDuplicable();
    AddComment(R"DOC(
Run graph by Gcu runtime.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(gcu_runtime, ops::GcuRuntimeOp, ops::GcuRuntimeOpMaker);

#endif  // PADDLE_WITH_GCU

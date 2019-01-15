// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <vector>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class PreparedKernel {
 public:
  PreparedKernel(const framework::OperatorBase& op,
                 const framework::RuntimeContext& ctx,
                 framework::OperatorWithKernel::OpKernelFunc func,
                 platform::DeviceContext* dev_ctx)
      : op(op), ctx(ctx), func(func), dev_ctx(dev_ctx), run(false) {}

  // These are copied from OperatorWithKernel::RunImpl, easier to customize for
  // inference.
  static std::unique_ptr<PreparedKernel> Prepare(
      const framework::RuntimeContext& ctx,
      const framework::OperatorWithKernel& op, const platform::Place& place) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto* dev_ctx = pool.Get(place);

    // check if op[type] has kernel registered.
    auto& all_op_kernels = op.AllOpKernels();
    auto kernels_iter = all_op_kernels.find(op.Type());
    if (kernels_iter == all_op_kernels.end()) {
      PADDLE_THROW(
          "There are no kernels which are registered in the %s operator.",
          op.Type());
    }

    framework::OperatorWithKernel::OpKernelMap& kernels = kernels_iter->second;

    auto expected_kernel_key = op.GetExpectedKernelType(
        framework::ExecutionContext(op, framework::Scope(), *dev_ctx, ctx));
    VLOG(3) << "expected_kernel_key:" << expected_kernel_key;

    auto kernel_iter = kernels.find(expected_kernel_key);
#ifdef PADDLE_WITH_MKLDNN
    // workaround for missing MKLDNN kernel when FLAGS_use_mkldnn env var is set
    if (kernel_iter == kernels.end() &&
        expected_kernel_key.library_type_ == framework::LibraryType::kMKLDNN) {
      VLOG(3) << "missing MKLDNN kernel: fallbacking to PLAIN one";
      expected_kernel_key.library_type_ = framework::LibraryType::kPlain;
      expected_kernel_key.data_layout_ = framework::DataLayout::kAnyLayout;
      kernel_iter = kernels.find(expected_kernel_key);
    }
#endif
    if (kernel_iter == kernels.end()) {
      PADDLE_THROW("op %s does not have kernel for %s", op.Type(),
                   KernelTypeToString(expected_kernel_key));
    }
    return std::unique_ptr<PreparedKernel>(
        new PreparedKernel(op, ctx, kernel_iter->second, dev_ctx));
  }

  const framework::OperatorBase& op;
  const framework::RuntimeContext ctx;
  framework::OperatorWithKernel::OpKernelFunc func;
  platform::DeviceContext* dev_ctx;
  bool run;
};

struct PreparedOp {
  framework::OperatorBase& op;
  const framework::RuntimeContext ctx;
  platform::DeviceContext* dev_ctx;
};

struct OpAndKernel {
  void Run(Scope* scope, platform::Place place) {
    if (is_kernel) {
      kernel->op.RuntimeInferShape(*scope, place, kernel->ctx);
      kernel->func(framework::ExecutionContext(kernel->op, *scope,
                                               *kernel->dev_ctx, kernel->ctx));
    } else {
      op->op.RuntimeInferShape(*scope, place, op->ctx);
      op->op.Run(*scope, place);
    }
  }
  std::unique_ptr<PreparedKernel> kernel;
  std::unique_ptr<PreparedOp> op;
  bool is_kernel{false};
};

/*
 * Simple, intuitive and effective. Only single thread is supported, and
 * currently designed for inference.
 */
class NaiveExecutor {
 public:
  explicit NaiveExecutor(const platform::Place& place) : place_(place) {}

  // Create child scope.
  // Create variables.
  // @with_feed_fetch_ops: whether to work with the feed and fetch operators.
  void Prepare(Scope* scope, const ProgramDesc& program_desc, int block_id,
               bool with_feed_fetch_ops);

  // Create variables before head.
  // Create parameters if persistable is ture, or create the temporary variables
  // instead.
  void CreateVariables(const ProgramDesc& desc, int block_id, bool persistable,
                       Scope* scope);

  // Run all the operators.
  void Run();

  // Get an tensor to operating directly, without the need for feed_ops.
  LoDTensor* FindTensor(const std::string& name);

  Scope* scope() { return scope_; }

  void CleanFeedFetchOps();

 protected:
  void CreateOps(const ProgramDesc& desc, int block_id,
                 bool with_feed_fetch_ops);

 private:
  const platform::Place place_;
  // Catch the required resource to avoid recreate.
  std::vector<std::unique_ptr<OperatorBase>> ops_;
  Scope* scope_;

  // Operator or Kernel cached Ops.
  std::vector<OpAndKernel> prepared_ops_;
  bool op_prepared_{false};
};

}  // namespace framework
}  // namespace paddle

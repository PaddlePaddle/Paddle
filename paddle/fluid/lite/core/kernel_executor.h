// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/lite/core/mir/ssa_graph.h"

namespace paddle {
namespace lite {

/*
 * KernelExecutor executes a list of kernels.
 */
class KernelExecutorBase {
 public:
  KernelExecutorBase(std::unique_ptr<mir::Program>&& program);

  // Prepare runtime context.
  void PrepareWorkspace();

  void Run();

 private:
  lite::Scope* scope_{};
  lite::Scope* exec_scope_{};
};

/*
 * KernelExecutor executes the kernels without concurrency, works in X86 place.
 */
class SerialKernelExecutor : public KernelExecutorBase {};

/*
 * KernelExecutor executes the kernels with CUDA like stream parallel support,
 * works in CUDA like devices.
 */
class StreamKernelExecutor : public KernelExecutorBase {};

}  // namespace lite
}  // namespace paddle

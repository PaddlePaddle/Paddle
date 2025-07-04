// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_CUDA || defined(PADDLE_WITH_HIP)

#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/execution_config.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/phi/backends/gpu/cuda/cuda_graph.h"

namespace ir {
class Operation;
}  // namespace ir

namespace paddle {
namespace framework {
class Scope;
class Value;
class PirInterpreter;
class ValueExecutionInfo;

class CudaGraphInstruction : public InstructionBase {
 public:
  CudaGraphInstruction(size_t id,
                       const phi::Place& place,
                       ::pir::Operation* op,
                       uint8_t* cuda_graph_state_ref,
                       int64_t cuda_graph_capture_pool_id,
                       ValueExecutionInfo* value_exe_info,
                       interpreter::ExecutionConfig execution_config);

  ~CudaGraphInstruction();

  void Run() override;

  const std::string& Name() const override { return name_; }

  ::pir::Operation* Operation() const override { return op_; }

  PirInterpreter* interpreter() const { return interpreter_; }

  void SetOutputHooks(const std::vector<PirHookFunc>& hookfuncs);

  void SetInputHooks(const std::vector<PirHookFunc>& hookfuncs);

 private:
  const phi::Place& place_;
  pir::Operation* op_;
  uint8_t* cuda_graph_state_ref_ = nullptr;
  int64_t cuda_graph_capture_pool_id_ = -1;

  std::string name_{"cuda_graph_instruction"};

  std::vector<Variable*> input_vars_;
  std::vector<Variable*> output_vars_;

  PirInterpreter* interpreter_ = nullptr;

  std::vector<std::string> skip_gc_names_;

  std::unique_ptr<phi::backends::gpu::CUDAGraph> cuda_graph_ = nullptr;
  std::vector<phi::DenseTensor> input_tensors_;
  std::vector<phi::DenseTensor> output_tensors_;
};

}  // namespace framework
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA

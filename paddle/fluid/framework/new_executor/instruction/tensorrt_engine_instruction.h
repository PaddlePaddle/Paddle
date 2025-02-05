// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/platform/tensorrt/engine.h"
#include "paddle/phi/core/platform/device_context.h"
namespace pir {
class Operation;
}  // namespace pir

namespace paddle {
namespace framework {
class Scope;
class ValueExecutionInfo;

class TensorRTEngineInstruction : public InstructionBase {
 public:
  TensorRTEngineInstruction(size_t id,
                            const phi::Place& place,
                            ::pir::Operation* op,
                            const ValueExecutionInfo* value_exec_info);

  ::pir::Operation* Operation() const override { return op_; }

  void Run() override;

  const std::string& Name() const override { return op_name_; }

 private:
  std::string ReadBinaryFileToString(const std::string& filePath);
  void PrepareDynamicShape();
  void RunTrt();
  void BindInputTensor(const std::string& input_name,
                       const phi::DenseTensor& input_tensor,
                       const Scope& scope,
                       std::vector<void*>& buffers,  // NOLINT
                       std::vector<int>& shape_v,    // NOLINT
                       int* runtime_batch);
  void BindOutputTensor(std::string output_name,
                        phi::DenseTensor* output_tensor,
                        int output_index,
                        std::vector<void*>& buffers,  // NOLINT
                        int* runtime_batch);
  std::unique_ptr<paddle::platform::TensorRTEngine> trt_engine_;  // not owned
  int64_t workspace_size_;
  bool allow_build_at_runtime_;
  std::unordered_map<int, std::string>
      input_names_;  // Only record input name that is not empty
  int input_nums_ = 0;
  std::unordered_map<int, std::string>
      output_names_;  // Only record output name that is not empty
  int output_nums_ = 0;
  std::vector<int> outputs_rank_;
  std::vector<phi::DataType> outputs_dtype_;
  std::string op_name_ = "pd_op.tensorrt_engine";
  ::pir::Operation* op_{nullptr};  // not owned

  const ValueExecutionInfo* value_exec_info_;  // not owned
};
}  // namespace framework
}  // namespace paddle
#endif

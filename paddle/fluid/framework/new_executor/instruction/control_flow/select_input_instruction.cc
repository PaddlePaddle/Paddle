// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/select_input_instruction.h"
#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"

namespace paddle {
namespace framework {

SelectInputInstruction::SelectInputInstruction(
    size_t id,
    const platform::Place &place,
    ::pir::Operation *op,
    ValueExecutionInfo *value_exe_info)
    : InstructionBase(id, place),
      op_(op),
      type_(OpFuncType::kCpuSync),
      inputs_() {
  VLOG(6) << "construct select_input instruction";

  std::unordered_map<pir::Value, std::vector<int>> inputs;
  mask_ = value_exe_info->GetVarByValue(op->operand_source(0));
  inputs.emplace(op->operand_source(0),
                 GetValueIds(op->operand_source(0), *value_exe_info));

  for (size_t i = 1; i < op->num_operands(); ++i) {
    inputs_.push_back(value_exe_info->GetVarByValue(op->operand_source(i)));
    inputs.emplace(op->operand_source(i),
                   GetValueIds(op->operand_source(i), *value_exe_info));
  }
  SetInputs(inputs);

  std::unordered_map<pir::Value, std::vector<int>> outputs;
  out_ = value_exe_info->GetVarByValue(op->result(0));
  outputs.emplace(op->result(0), GetValueIds(op->result(0), *value_exe_info));
  SetOutputs(outputs);
}

inline int GetBranchNumber(const phi::DenseTensor &mask) {
  PADDLE_ENFORCE_EQ(
      mask.numel(),
      1,
      phi::errors::Fatal("The numel of Input(Mask) in SelectInputOp or "
                         "SelectOutputOp must be 1. "
                         "But received %d, and it's shape is [%s].",
                         mask.numel(),
                         mask.dims()));
  if (platform::is_cpu_place(mask.place())) {
    return mask.data<int>()[0];
  }
  // when platform::is_gpu_place(mask.place()) is true
  std::unique_ptr<phi::DenseTensor> cpu_mask{new phi::DenseTensor()};
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_CUSTOM_DEVICE) || defined(PADDLE_WITH_XPU)
  framework::TensorCopySync(mask, phi::CPUPlace(), cpu_mask.get());
#else
  PADDLE_THROW(phi::errors::Fatal(
      "This version of PaddlePaddle does NOT support GPU, "
      "but got GPU tensor 'Mask' in SelectInputOp or SelectOutputOp. "
      "Please compile PaddlePaddle WITH_GPU first."));
#endif
  return cpu_mask->data<int>()[0];
}

class AssignFunctor {
 public:
  explicit AssignFunctor(Variable *out) : out_(out) {}

  void operator()(const phi::DenseTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<phi::DenseTensor>();
    copy_tensor(lod_tensor, &out_tensor);
  }

  void operator()(const phi::TensorArray &array) const {
    auto &out_array = *out_->GetMutable<phi::TensorArray>();
    out_array.resize(array.size());
    for (size_t i = 0; i < array.size(); ++i) {
      copy_tensor(array[i], &out_array[i]);
    }
  }

  void operator()(const phi::SelectedRows &rows) const {
    phi::SelectedRows &out_rows = *out_->GetMutable<phi::SelectedRows>();
    out_rows.set_rows(rows.rows());
    out_rows.set_height(rows.height());
    auto &t = rows.value();
    auto *m = out_rows.mutable_value();
    TensorCopy(t, t.place(), m);
  }

  template <typename T>
  void operator()(const T &v UNUSED) const {
    PADDLE_ENFORCE_EQ(
        true,
        false,
        platform::errors::PermissionDenied(
            "Not support type for assign op with type %s", typeid(T).name()));
  }

 private:
  void copy_tensor(const phi::DenseTensor &lod_tensor,
                   phi::DenseTensor *out) const {
    if (!lod_tensor.IsInitialized()) return;
    auto &out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  Variable *out_;
};

void SelectInputInstruction::Run() {
  VLOG(6) << "run select_input instruction";
  auto &mask = mask_->Get<phi::DenseTensor>();
  size_t output_branch = static_cast<size_t>(GetBranchNumber(mask));
  PADDLE_ENFORCE_LT(
      output_branch,
      inputs_.size(),
      phi::errors::Fatal(
          "Input 'Mask' in SelectInputOp is invalid. "
          "'Mask' must be less than the size of input vector 'X'. "
          "But received Mask = %d, X's size = %d.",
          output_branch,
          inputs_.size()));
  Variable *selected = inputs_[output_branch];
  VisitVarType(*selected, AssignFunctor(out_));
}

}  // namespace framework
}  // namespace paddle

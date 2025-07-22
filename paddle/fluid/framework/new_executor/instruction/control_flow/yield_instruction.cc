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

#include "paddle/fluid/framework/new_executor/instruction/control_flow/yield_instruction.h"

#include "paddle/fluid/framework/new_executor/instruction/instruction_util.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/pir/include/core/builtin_type.h"

COMMON_DECLARE_bool(check_cuda_error);

namespace paddle {
namespace framework {

YieldInstruction::YieldInstruction(size_t id,
                                   const phi::Place &place,
                                   ::pir::Operation *op,
                                   ValueExecutionInfo *value_exe_info)
    : InstructionBase(id, place), op_(op), value_exe_info_(value_exe_info) {
  VLOG(6) << "construct yield instruction";
  auto parent_op = op->GetParentOp();
  std::unordered_map<pir::Value, std::vector<int>> inputs;
  for (size_t i = 0; i < op->num_operands(); ++i) {
    // Skip the first input (cond) when the parent op is a while op.
    if (parent_op->isa<paddle::dialect::WhileOp>() && i == 0) {
      continue;
    }
    auto in = op->operand_source(i);
    if (in) {
      inputs.emplace(in, GetValueIds(in, *value_exe_info_));
      input_vars_.push_back(value_exe_info_->GetVarByValue(in));
    } else {
      // value 为空的时候根据 parent op 输出 value 的 meta 信息填一个全 0
      // tensor。Build instruction 的时候先创建 var
      if (parent_op->result(i) && parent_op->result(i).type()) {
        auto out_type = parent_op->result(i).type();
        std::string new_name = "_fake_var_op_" + std::to_string(op->id()) +
                               "_input_" + std::to_string(i) + "_";
        Variable *fake_var = value_exe_info_->GetScope()->Var(new_name);
        if (out_type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
          fake_var->GetMutable<phi::DenseTensor>();
          input_vars_.push_back(fake_var);
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "unsupported type %d", out_type.dyn_cast<pir::Type>().type_id()));
        }
      }
    }
  }
  SetInputs(inputs);

  for (size_t i = 0; i < parent_op->num_results(); ++i) {
    if (parent_op->result(i) && parent_op->result(i).type()) {
      output_vars_.push_back(
          value_exe_info->GetVarByValue(parent_op->result(i)));
    }
  }

  PADDLE_ENFORCE_EQ(
      input_vars_.size(),
      output_vars_.size(),
      common::errors::InvalidArgument("The number of inputs in YieldOp and "
                                      "outputs of parent op must be equal."
                                      "But received %d and %d.",
                                      input_vars_.size(),
                                      output_vars_.size()));
}

template <typename T0, typename T1>
void FullFakeTensor(const pir::Value &output_value, Variable *output_var) {
  if (!output_value || !output_var) {
    output_var = nullptr;
    return;
  }
  auto out_tensor_type = output_value.type().dyn_cast<T0>();
  auto abs_dims = out_tensor_type.dims();
  for (int i = 0; i < abs_dims.size(); ++i) {
    // dynamic shape, set to 1
    if (abs_dims[i] == -1) {
      abs_dims[i] = 1;
    }
  }
#ifdef PADDLE_WITH_CUDA
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  auto *dev_ctx = pool.Get(phi::GPUPlace());
  phi::DataType dtype =
      paddle::dialect::TransToPhiDataType(out_tensor_type.dtype());
  phi::FullKernel<float, phi::GPUContext>(
      *(static_cast<phi::GPUContext *>(dev_ctx)),
      phi::IntArray(common::vectorize(abs_dims)),
      0.0,
      dtype,
      output_var->GetMutable<T1>());
#else
  VLOG(1) << "unsupported Device for Fake Tensor.";
  output_var = nullptr;
  return;
#endif
}
void YieldInstruction::Run() {
  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("YieldInstruction begin");
  }

  for (size_t i = 0; i < input_vars_.size(); ++i) {
    if (input_vars_[i] == nullptr) {
      output_vars_[i] = nullptr;
    } else if (input_vars_[i]->IsType<phi::DenseTensor>()) {
      if (input_vars_[i]->IsInitialized() &&
          !input_vars_[i]->Get<phi::DenseTensor>().initialized()) {
        // 对应 input 为 NULL VALUE 的情况，fake tensor
        FullFakeTensor<paddle::dialect::AllocatedDenseTensorType,
                       phi::DenseTensor>(
            value_exe_info_->GetValueByVar(output_vars_[i]), output_vars_[i]);
      } else {
        output_vars_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
            input_vars_[i]->Get<phi::DenseTensor>());
      }
    } else if (input_vars_[i]->IsType<phi::TensorArray>()) {
      const auto &inner_array = input_vars_[i]->Get<phi::TensorArray>();
      auto *output_array = output_vars_[i]->GetMutable<phi::TensorArray>();
      *output_array = inner_array;
    } else {
      PADDLE_THROW(common::errors::Unimplemented("unsupported type %d",
                                                 input_vars_[i]->Type()));
    }
  }
  if (FLAGS_check_cuda_error) [[unlikely]] {
    CUDAErrorCheck("YieldInstruction finish");
  }
}

}  // namespace framework
}  // namespace paddle

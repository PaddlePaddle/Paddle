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

namespace paddle {
namespace framework {

static double GetDenseTensorEleSum(const phi::DenseTensor &tensor) {
  phi::DenseTensor cpu_tensor;
  phi::CPUPlace place;
  paddle::framework::TensorCopy(tensor, place, &cpu_tensor);
  phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(tensor.place());
  dev_ctx.Wait();
  double sum = 0.0;
  for (int64_t i = 0; i < cpu_tensor.numel(); i++) {
    if (cpu_tensor.dtype() == phi::DataType::FLOAT32) {
      sum += static_cast<double>(cpu_tensor.data<float>()[i]);
    } else if (cpu_tensor.dtype() == phi::DataType::FLOAT64) {
      sum += static_cast<double>(cpu_tensor.data<double>()[i]);
    } else if (cpu_tensor.dtype() == phi::DataType::INT32) {
      sum += static_cast<double>(cpu_tensor.data<int32_t>()[i]);
    } else if (cpu_tensor.dtype() == phi::DataType::INT64) {
      sum += static_cast<double>(cpu_tensor.data<int64_t>()[i]);
    } else if (cpu_tensor.dtype() == phi::DataType::FLOAT16) {
      const phi::dtype::float16 *data = cpu_tensor.data<phi::dtype::float16>();
      sum += static_cast<double>(data[0]);
    } else if (cpu_tensor.dtype() == phi::DataType::BOOL) {
      sum += static_cast<double>(cpu_tensor.data<bool>()[i]);
    } else {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  return sum;
}
YieldInstruction::YieldInstruction(size_t id,
                                   const phi::Place &place,
                                   ::pir::Operation *op,
                                   ValueExecutionInfo *value_exe_info)
    : InstructionBase(id, place), op_(op) {
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
      inputs.emplace(in, GetValueIds(in, *value_exe_info));
      input_vars_.push_back(value_exe_info->GetVarByValue(in));
    } else {
      // value 为空的时候根据 parent op 的输出 value，填一个全 0 tensor进去
      if (parent_op->result(i) && parent_op->result(i).type()) {
#ifdef PADDLE_WITH_CUDA
        auto out_type = parent_op->result(i).type();
        if (out_type.isa<paddle::dialect::AllocatedDenseTensorType>()) {
          auto out_densetensor_type =
              out_type.dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
          auto abs_dims = out_densetensor_type.dims();
          for (int i = 0; i < abs_dims.size(); ++i) {
            if (abs_dims[i] == -1) {
              abs_dims[i] = 1;
            }
          }
          const auto GetNumElementsFromDim =
              [](const ::pir::DDim &dim) -> int64_t {
            return ::common::product(dim);
          };
          int64_t numel = GetNumElementsFromDim(abs_dims);
          std::string new_name = "_fake_var_op_" + std::to_string(op->id()) +
                                 "_input_" + std::to_string(i) + "_";
          Variable *fake_var = value_exe_info->GetScope()->Var(new_name);

          phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
          auto *dev_ctx = pool.Get(phi::GPUPlace());
          phi::DataType dtype =
              paddle::dialect::TransToPhiDataType(out_densetensor_type.dtype());
          phi::FullKernel<float, phi::GPUContext>(
              *(static_cast<phi::GPUContext *>(dev_ctx)),
              phi::IntArray(common::vectorize(abs_dims)),
              0.0,
              dtype,
              fake_var->GetMutable<phi::DenseTensor>());
          input_vars_.push_back(fake_var);
          eager_gc_input_var_idxs_.push_back(i);
        } else {
          PADDLE_THROW(common::errors::Unimplemented(
              "unsupported type %d", out_type.dyn_cast<pir::Type>().type_id()));
        }
#else
        input_vars_.push_back(nullptr);
#endif
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

void YieldInstruction::Run() {
  for (size_t i = 0; i < input_vars_.size(); ++i) {
    if (input_vars_[i] == nullptr) {
      output_vars_[i] = nullptr;
    } else if (input_vars_[i]->IsType<phi::DenseTensor>()) {
      output_vars_[i]->GetMutable<phi::DenseTensor>()->ShareDataWith(
          input_vars_[i]->Get<phi::DenseTensor>());
    } else if (input_vars_[i]->IsType<phi::TensorArray>()) {
      const auto &inner_array = input_vars_[i]->Get<phi::TensorArray>();
      auto *output_array = output_vars_[i]->GetMutable<phi::TensorArray>();
      *output_array = inner_array;
    } else {
      PADDLE_THROW(common::errors::Unimplemented("unsupported type %d",
                                                 input_vars_[i]->Type()));
    }
  }
  for (auto idx : eager_gc_input_var_idxs_) {
    AddEagerGCVar(input_vars_[idx]);
  }
}

}  // namespace framework
}  // namespace paddle

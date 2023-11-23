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

#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"

#include <cuda_runtime.h>
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"

namespace paddle {
namespace framework {

typedef void (*lower_func_ptr_g)(void*, int32_t, void*);

class CinnJitInstruction::FnPtrImpl {
  using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

 public:
  explicit FnPtrImpl(const CINNKernelInfo& cuda_jit_info)
      : cuda_jit_info_(cuda_jit_info) {}

  void Run(const std::vector<phi::DenseTensor*>& kernel_args, void* stream) {
    func_args_.clear();
    // parse args'data of tensor
    for (size_t i = 0; i < kernel_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      buffer->memory = reinterpret_cast<uint8_t*>(kernel_args[i]->data());
      func_args_.emplace_back(buffer);
    }
    // parse arg's data about tensor shape
    for (const auto& int_arg_mp : cuda_jit_info_.int_args_map) {
      func_args_.emplace_back(kernel_args[int_arg_mp.second.arg_idx]->dims().at(
          int_arg_mp.second.dim_idx));
      VLOG(1) << "DEBUG size = "
              << kernel_args[int_arg_mp.second.arg_idx]->dims().at(
                     int_arg_mp.second.dim_idx);
    }

    // launch host kernel
    ((lower_func_ptr_g)cuda_jit_info_.fn_ptr)(
        static_cast<void*>(func_args_.data()), func_args_.size(), stream);
  }

 private:
  CINNKernelInfo cuda_jit_info_;

  std::vector<cinn_pod_value_t> func_args_;
};

CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place) {
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cinn_kernel_info());
  op_ = op;

  place_ = place;

  InitInputsOutputsIds(op, value_exec_info);

  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);

    auto var_name = value_exec_info.GetVarName(in);
    Scope* inner_scope = value_exec_info.GetScope();
    auto var = inner_scope->FindVar(var_name);
    const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());

    auto tensor = value_exec_info.GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);
  }

  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value result = op->result(i);
    auto var_name = value_exec_info.GetVarName(result);

    auto tensor = value_exec_info.GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);

    // out_tensor_ = tensor;
    //
    // auto alloc_tensor_type =
    //     result.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    // tensor->set_type(
    //     paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    // tensor->Resize(alloc_tensor_type.dims());
  }
}

void CinnJitInstruction::Run() {
  auto gpu_ctx = static_cast<phi::GPUContext*>(dev_ctx_);
  auto stream = gpu_ctx->stream();
  VLOG(1) << "DEBUG tensor_args_.size() = " << tensor_args_.size();
  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    tensor_args_[i]->Resize(tensor_args_[0]->dims());
    gpu_ctx->Alloc(tensor_args_[i], tensor_args_[i]->dtype());
    VLOG(1) << "DEBUG tensor_args[" << i
            << "].size() = " << tensor_args_[i]->dims().at(0);
  }
  // fn_ptr_impl_->Run(tensor_args_, static_cast<void*>(stream));
  fn_ptr_impl_->Run(tensor_args_, nullptr);
}

const std::string& CinnJitInstruction::Name() const {
  static const std::string name = "cinn_jit";
  return name;
}

}  // namespace framework
}  // namespace paddle

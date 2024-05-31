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

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/pir_compiler.h"
#include "paddle/common/errors.h"
#include "paddle/common/performance_statistician.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/cinn/runtime/cinn_runtime.h"
#endif
PD_DECLARE_bool(cinn_bucket_compile);
PD_DECLARE_bool(cinn_enable_config_search);
PD_DECLARE_string(cinn_kernel_execution_label);

namespace paddle {
namespace framework {

typedef void (*lower_func_ptr_g)(void*, int32_t, void*);
typedef void (*infer_shape_func_ptr_g)(void*, int32_t, int64_t**);

class CinnJitInstruction::FnPtrImpl {
  using CINNKernelInfo = cinn::hlir::framework::pir::CINNKernelInfo;

 public:
  explicit FnPtrImpl(const CINNKernelInfo& cinn_kernel_info)
      : cinn_kernel_info_(cinn_kernel_info) {}

  void Run(const std::vector<phi::DenseTensor*>& kernel_args, void* stream) {
    VLOG(6) << "Start Run: " << cinn_kernel_info_.fn_name;
    func_args_.clear();

    // 1. Convert the phi::DenseTensor type to cinn_pod_value_t
    for (size_t i = 0; i < kernel_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      buffer->memory = reinterpret_cast<uint8_t*>(kernel_args[i]->data());
      func_args_.emplace_back(buffer);
    }
    // 2. Convert arg's data about shape of Tensor to cinn_pod_value_t
    for (const auto& int_arg_mp : cinn_kernel_info_.int_args_map) {
      func_args_.emplace_back(static_cast<int64_t>(
          kernel_args[int_arg_mp.second.arg_idx]->dims().at(
              int_arg_mp.second.dim_idx)));
    }

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Run func_args_ size: " << func_args_.size();
      for (const auto& args : func_args_) {
        VLOG(4) << " args type_code: " << args.type_code();
      }
    }

    // 3. Launch host kernel
    ((lower_func_ptr_g)cinn_kernel_info_.fn_ptr)(
        static_cast<void*>(func_args_.data()), func_args_.size(), stream);
    VLOG(6) << "End Run: " << cinn_kernel_info_.fn_name;
  }

  void InferShape(const std::vector<phi::DenseTensor*>& kernel_args,
                  int32_t input_tensor_size,
                  int32_t output_tensor_size) {
    VLOG(6) << "Start InferShape: " << cinn_kernel_info_.fn_name;
    func_args_.clear();

    // 1. Convert the phi::DenseTensor type to cinn_pod_value_t
    for (size_t i = 0; i < kernel_args.size(); ++i) {
      auto* buffer = new cinn_buffer_t();
      func_args_.emplace_back(buffer);
    }

    // 2. Convert arg's data about shape of Tensor to cinn_pod_value_t
    for (const auto& int_arg_mp : cinn_kernel_info_.int_args_map) {
      func_args_.emplace_back(static_cast<int64_t>(
          kernel_args[int_arg_mp.second.arg_idx]->dims().at(
              int_arg_mp.second.dim_idx)));
    }

    // 3. Define an array of Pointers to hold the output tensor shape
    std::vector<int64_t*> output_tensor_shapes(output_tensor_size);
    for (int i = 0; i < output_tensor_size; ++i) {
      output_tensor_shapes[i] = reinterpret_cast<int64_t*>(
          malloc(kernel_args[input_tensor_size + i]->dims().size() *
                 sizeof(int64_t*)));
    }

    if (VLOG_IS_ON(4)) {
      VLOG(4) << "InferShape func_args_ size: " << func_args_.size();
      for (const auto& args : func_args_) {
        VLOG(4) << " args type_code: " << args.type_code();
      }
    }

    // 4. Launch infer_shape_fn_ptr to infer shape of output tensor
    ((infer_shape_func_ptr_g)cinn_kernel_info_.infer_shape_fn_ptr)(
        static_cast<void*>(func_args_.data()),
        func_args_.size(),
        output_tensor_shapes.data());

    // 5. Resize shape of output tensor
    for (int i = 0; i < output_tensor_size; ++i) {
      DDim dim(output_tensor_shapes[i],
               kernel_args[input_tensor_size + i]->dims().size());
      kernel_args[input_tensor_size + i]->Resize(dim);
      free(output_tensor_shapes[i]);
    }
    VLOG(6) << "End InferShape: " << cinn_kernel_info_.fn_name;
  }

 private:
  CINNKernelInfo cinn_kernel_info_;

  std::vector<cinn_pod_value_t> func_args_;
};

CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo* value_exec_info)
    : InstructionBase(id, place) {
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cinn_kernel_info());
  op_ = op;
  input_tensor_size = op->num_operands();
  output_tensor_size = op->num_results();

  place_ = place;

  InitInputsOutputsIds(op, *value_exec_info);

  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);

    auto var_name = value_exec_info->GetVarName(in);
    auto tensor = value_exec_info->GetScope()
                      ->FindVar(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);
  }

  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value result = op->result(i);
    auto var_name = value_exec_info->GetVarName(result);

    auto tensor = value_exec_info->GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args_.push_back(tensor);
    auto alloc_tensor_type =
        result.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    tensor->set_type(
        paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    for (size_t j = 0; j < alloc_tensor_type.dims().size(); ++j) {
      if (alloc_tensor_type.dims()[j] < 0) {
        need_update_shape = true;
        continue;
      }
    }
    tensor->Resize(alloc_tensor_type.dims());
  }
}

void CinnJitInstruction::Run() {
#if defined(PADDLE_WITH_CUDA)
  auto gpu_ctx = static_cast<phi::GPUContext*>(dev_ctx_);

  auto stream = gpu_ctx->stream();

  if (FLAGS_cinn_bucket_compile && need_update_shape) {
    fn_ptr_impl_->InferShape(
        tensor_args_, input_tensor_size, output_tensor_size);
  }
  for (size_t i = 0; i < tensor_args_.size(); ++i) {
    gpu_ctx->Alloc(tensor_args_[i], tensor_args_[i]->dtype());
  }

  // 2. exexute kernel
  if (FLAGS_cinn_enable_config_search) {
    ::common::PerformanceStatistician& ps =
        ::common::PerformanceStatistician::Instance();
    ps.Start(FLAGS_cinn_kernel_execution_label);
    fn_ptr_impl_->Run(tensor_args_, static_cast<void*>(stream));
    cudaDeviceSynchronize();
    ps.End(FLAGS_cinn_kernel_execution_label);
  } else {
    fn_ptr_impl_->Run(tensor_args_, static_cast<void*>(stream));
  }
#else
  VLOG(0) << "Not Supported: cinn jit instruction currently does not "
             "support non-CUDA kernel";
#endif
}

const std::string& CinnJitInstruction::Name() const {
  static const std::string name = "cinn_jit";
  return name;
}

}  // namespace framework
}  // namespace paddle

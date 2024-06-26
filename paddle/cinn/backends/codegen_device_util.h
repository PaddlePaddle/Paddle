// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include <absl/container/flat_hash_map.h>

#include <string>
#include <tuple>
#include <vector>
#ifdef CINN_WITH_CUDA
#include "paddle/cinn/backends/codegen_cuda_dev.h"
#endif
#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/cinn/runtime/flags.h"
#include "paddle/common/enforce.h"
namespace cinn {
namespace backends {

#define KERNEL_ARGS "kernel_args"
#define KERNEL_ARGS_NUM "kernel_args_num"
#define KERNEL_STREAM "kernel_stream"
#define TENSOR_SHAPE_ARGS "tensor_shape_args"

/**
 * Split a CINN Module into two separate modules, one contains the host
 * functions, the other contains the device kernels.
 *
 * This contains some process:
 *
 * - replace the original kernel function with a Call node and add it to the
 * first module, add a device kernel function to the second module.
 */
std::tuple<ir::Module, ir::Module> SplitDeviceAndHostModule(ir::Module module);

namespace detail {

struct CollectHostFunctionVisitor : public ir::IRMutator<> {
  explicit CollectHostFunctionVisitor(const std::string& module_name)
      : host_module_builder(module_name + "_host",
                            cinn::common::DefaultHostTarget()),
        device_module_builder(module_name + "_gpu_device",
                              cinn::common::DefaultDeviceTarget()) {}

  std::tuple<ir::Module, ir::Module> operator()(Expr* expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return std::make_tuple(host_module_builder.Build(),
                           device_module_builder.Build());
  }

 protected:
  void Visit(const ir::_LoweredFunc_* op, Expr* expr) override {
    if (op->body.As<ir::Call>()) {
      host_module_builder.AddFunctionWithoutOptim(expr->as_lowered_func_ref());
    } else {
      if (!op->cuda_axis_info.valid()) {
        expr->as_lowered_func_ref()->cuda_axis_info.set_valid(true);
      }
      auto host_func =
          CreateHostFunctionGivenDeviceKernel(expr->as_lowered_func());
      host_module_builder.AddFunctionWithoutOptim(
          host_func.as_lowered_func_ref());

      device_module_builder.AddFunctionWithoutOptim(
          CreateDeviceFunctionGivenDeviceKernel(*expr).as_lowered_func_ref());
    }
  }

  /**
   * Create a wrapper function for a kernel.
   *
   * For example, we get a kernel function:
   *
   * \code
   * __global__
   * void fn (float* a, float* out) { ... }
   * \endcode
   *
   * A host wrapper function will generate for it
   *
   * \code
   * void fn (cinn_buffer_t* a, cinn_buffer_t* out) {
   *   Call(fn_kernel);
   * }
   * \endcode
   */
  Expr CreateHostFunctionGivenDeviceKernel(ir::_LoweredFunc_* func) {
    // std::vector<Expr> args;
    // NOTE the suffix `__ptr` makes this argument lower to a pointer in LLVM
    // backend. args.push_back(Var("args__ptr", type_of<cinn_pod_value_t*>()));
    // args.push_back(Var("num_args", type_of<int32_t>()));
    ir::Var kernel_ptr(GenDeviceKernelName(func->name), type_of<std::string>());
    ir::Var kernel_args(KERNEL_ARGS, type_of<void*>());
    ir::Var kernel_args_num(KERNEL_ARGS_NUM, type_of<int>());
    ir::Var kernel_stream(KERNEL_STREAM, type_of<void*>());

    // shared_mem_bytes Can be calculated after codegen_cuda_dev buffer creation
    // however, this make CodeGenCUDA_Dev before spliting the host and device
    // module Maybe we could reorder the process.
    std::optional<Expr> shared_mem_bytes;
    cinn::common::DefaultDeviceTarget().arch.Match(
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
#ifdef CINN_WITH_CUDA
          CodeGenCUDA_Dev codegen_dev(cinn::common::DefaultNVGPUTarget());
          codegen_dev.Compile(ir::LoweredFunc(func));
          shared_mem_bytes = codegen_dev.GetDynSharedMemOffset();
#endif
        },
        [&](common::HygonDCUArchHIP) {
          PADDLE_THROW(phi::errors::Unimplemented(
              "CINN todo: new hardware HygonDCUArchHIP"));
        });

    VLOG(6) << "Add a call node for func->name " << func->name << "\n"
            << "grid_dim: (" << func->cuda_axis_info.grid_dim(0) << ", "
            << func->cuda_axis_info.grid_dim(1) << ", "
            << func->cuda_axis_info.grid_dim(2) << "), "
            << "block_dim: (" << func->cuda_axis_info.block_dim(0) << ", "
            << func->cuda_axis_info.block_dim(1) << ", "
            << func->cuda_axis_info.block_dim(2) << "), "
            << "shared_mem: " << shared_mem_bytes.value();

    std::optional<const char*> call_kernel;
    cinn::common::DefaultDeviceTarget().arch.Match(
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) { CINN_NOT_IMPLEMENTED; },
        [&](common::NVGPUArch) {
          call_kernel = runtime::intrinsic::call_cuda_kernel;
        },
        [&](common::HygonDCUArchHIP) {
          PADDLE_THROW(phi::errors::Unimplemented(
              "CINN todo: new hardware HygonDCUArchHIP"));
        });

    auto call_extern_api =
        ir::Call::Make(Void(),
                       call_kernel.value(),
                       {kernel_ptr,
                        kernel_args,
                        kernel_args_num,
                        func->cuda_axis_info.grid_dim(0),   // grid_x
                        func->cuda_axis_info.grid_dim(1),   // grid_y
                        func->cuda_axis_info.grid_dim(2),   // grid_z
                        func->cuda_axis_info.block_dim(0),  // block_x
                        func->cuda_axis_info.block_dim(1),  // block_y
                        func->cuda_axis_info.block_dim(2),  // block_z
                        shared_mem_bytes.value(),
                        kernel_stream},
                       {},
                       ir::CallType::Extern,
                       ir::FunctionRef(),
                       0);
    std::vector<ir::Argument> arguments = {
        ir::Argument(kernel_args, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num, ir::Argument::IO::kInput),
        ir::Argument(kernel_stream, ir::Argument::IO::kOutput)};

    return ir::_LoweredFunc_::Make(func->name, arguments, call_extern_api, {});
  }

  Expr CreateDeviceFunctionGivenDeviceKernel(Expr expr) {
    auto copied = ir::ir_utils::IRCopy(expr);
    auto* lowered_func = copied.as_lowered_func();
    lowered_func->name = GenDeviceKernelName(lowered_func->name);
    return copied;
  }

  inline std::string GenDeviceKernelName(const std::string& fn) {
    return fn + "_kernel";
  }

 protected:
  ir::Module::Builder host_module_builder;
  ir::Module::Builder device_module_builder;
};

struct CollectBucketStrategyHostFunctionVisitor
    : public CollectHostFunctionVisitor {
  explicit CollectBucketStrategyHostFunctionVisitor(
      const std::string& module_name)
      : CollectHostFunctionVisitor(module_name),
        kernel_args_(KERNEL_ARGS, type_of<void*>()),
        kernel_args_num_(KERNEL_ARGS_NUM, type_of<int>()),
        kernel_stream_(KERNEL_STREAM, type_of<void*>()),
        tensor_shape_args_(TENSOR_SHAPE_ARGS, type_of<int64_t**>()) {}

  std::tuple<ir::Module, ir::Module> operator()(Expr* expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return std::make_tuple(host_module_builder.Build(),
                           device_module_builder.Build());
  }

 private:
  static bool compare_priority(const std::pair<int, std::pair<Expr, Expr>>& a,
                               const std::pair<int, std::pair<Expr, Expr>>& b) {
    return a.first > b.first;
  }
  void Visit(const ir::_Module_* op, Expr* expr) {
    if (op->functions.size() == 1 && op->predicates.size() == 0) {
      expr->as_module()->predicates.push_back(ir::Expr(true));
    }
    PADDLE_ENFORCE_EQ(
        op->functions.size(),
        op->predicates.size(),
        phi::errors::InvalidArgument(
            "The size of functions and predicates should be equal"));
    PADDLE_ENFORCE_EQ(
        op->functions.size(),
        op->priorities.size(),
        phi::errors::InvalidArgument(
            "The size of functions and priorities should be equal"));
    // Sort funcitons and predicates according to the priority
    std::vector<std::pair<Expr, Expr>> func_predicate;
    std::vector<std::pair<int, std::pair<Expr, Expr>>> predicate_priority;
    VLOG(3) << "The number of the functions is " << op->functions.size();
    for (int i = 0; i < op->functions.size(); i++) {
      auto func_pair = std::make_pair(op->functions[i], op->predicates[i]);
      func_predicate.push_back(func_pair);
      predicate_priority.push_back(
          std::make_pair(op->priorities[i], func_pair));
    }
    sort(
        predicate_priority.begin(), predicate_priority.end(), compare_priority);
    predicate_priority[0].second.first;

    for (int i = 0; i < op->functions.size(); ++i) {
      ProcessLoweredFunc(predicate_priority[i].second.first,
                         predicate_priority[i].second.second);
      if (i == 0) {
        ProcessArgs(op->functions[i]);
      }
    }

    std::vector<ir::Argument> arguments = {
        ir::Argument(kernel_args_, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num_, ir::Argument::IO::kInput),
        ir::Argument(kernel_stream_, ir::Argument::IO::kOutput)};
    std::vector<ir::Expr> body_stmts(arg_defs_);
    body_stmts.insert(body_stmts.end(), buckets_.begin(), buckets_.end());
    ir::Expr host_func =
        ir::_LoweredFunc_::Make(op->functions[0].as_lowered_func()->name,
                                arguments,
                                ir::Block::Make(body_stmts),
                                {});
    host_module_builder.AddFunctionWithoutOptim(
        host_func.as_lowered_func_ref());

    // Parse LoweredFunc to infer output tensor's shape
    std::vector<ir::Expr> infer_shape_func_body_stmts(arg_defs_);
    infer_shape_func_body_stmts.insert(
        infer_shape_func_body_stmts.end(),
        op->infer_shape_func.as_lowered_func()->body);

    std::vector<ir::Argument> infer_shape_arguments = {
        ir::Argument(kernel_args_, ir::Argument::IO::kOutput),
        ir::Argument(kernel_args_num_, ir::Argument::IO::kInput),
        ir::Argument(tensor_shape_args_, ir::Argument::IO::kOutput)};

    ir::Expr host_infer_shape_func =
        ir::_LoweredFunc_::Make(op->infer_shape_func.as_lowered_func()->name,
                                infer_shape_arguments,
                                ir::Block::Make(infer_shape_func_body_stmts),
                                {});
    host_module_builder.AddFunctionWithoutOptim(
        host_infer_shape_func.as_lowered_func_ref());
  }

  void ProcessLoweredFunc(ir::Expr func, ir::Expr predicate);

  void ProcessArgs(ir::Expr func);

  Expr CreateDeviceFunction(ir::Expr expr, ir::Expr predicate);

  inline std::string GenDeviceKernelName(const std::string& fn_name,
                                         ir::Expr predicate);

 private:
  std::vector<ir::Expr> buckets_;
  std::vector<ir::Expr> arg_defs_;

  ir::Var kernel_args_;
  ir::Var kernel_args_num_;
  ir::Var kernel_stream_;
  ir::Var tensor_shape_args_;
};

}  // namespace detail

}  // namespace backends
}  // namespace cinn

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

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/ir/ir.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/ir/utils/ir_copy.h"

namespace cinn {
namespace backends {

#define KERNEL_ARGS "kernel_args"
#define KERNEL_ARGS_NUM "kernel_args_num"
#define KERNEL_STREAM "kernel_stream"

/**
 * Split a CINN Module into two separate modules, one cantains the host
 * functions, the other contains the device kernels.
 *
 * This contains some process:
 *
 * - replace the original kernel function with a Call node and add it to the
 * first module, add a device kernel function to the second module.
 */
std::tuple<ir::Module, ir::Module> SplitCudaAndHostModule(ir::Module module);

namespace detail {

struct CollectHostFunctionVisitor : public ir::IRMutator<> {
  explicit CollectHostFunctionVisitor(const std::string& module_name)
      : host_module_builder(module_name + "_host", common::DefaultHostTarget()),
        device_module_builder(module_name + "_gpu_device",
                              common::DefaultNVGPUTarget()) {}

  std::tuple<ir::Module, ir::Module> operator()(Expr* expr) {
    ir::IRMutator<>::Visit(expr, expr);
    return std::make_tuple(host_module_builder.Build(),
                           device_module_builder.Build());
  }

 private:
  void Visit(const ir::_LoweredFunc_* op, Expr* expr) override {
    if (op->body.As<ir::Call>()) {
      host_module_builder.AddFunctionWithoutOptim(expr->as_lowered_func_ref());
    } else {
      if (!op->cuda_axis_info.valid()) {
        expr->as_lowered_func_ref()->cuda_axis_info.set_valid(true);
      }
      auto host_func = CreateHostFunctionGivenDeviceKernel(op);
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
  Expr CreateHostFunctionGivenDeviceKernel(const ir::_LoweredFunc_* func) {
    // std::vector<Expr> args;
    // NOTE the suffix `__ptr` makes this argument lower to a pointer in LLVM
    // backend. args.push_back(Var("args__ptr", type_of<cinn_pod_value_t*>()));
    // args.push_back(Var("num_args", type_of<int32_t>()));
    ir::Var kernel_ptr(GenDeviceKernelName(func->name), type_of<std::string>());
    ir::Var kernel_args(KERNEL_ARGS, type_of<void*>());
    ir::Var kernel_args_num(KERNEL_ARGS_NUM, type_of<int>());
    ir::Var kernel_stream(KERNEL_STREAM, type_of<void*>());

    auto call_extern_api =
        ir::Call::Make(Void(),
                       runtime::intrinsic::call_cuda_kernel,
                       {kernel_ptr,
                        kernel_args,
                        kernel_args_num,
                        Expr(func->cuda_axis_info.grid_dim(0)),   // grid_x
                        Expr(func->cuda_axis_info.grid_dim(1)),   // grid_y
                        Expr(func->cuda_axis_info.grid_dim(2)),   // grid_z
                        Expr(func->cuda_axis_info.block_dim(0)),  // block_x
                        Expr(func->cuda_axis_info.block_dim(1)),  // block_y
                        Expr(func->cuda_axis_info.block_dim(2)),  // block_z
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

 private:
  ir::Module::Builder host_module_builder;
  ir::Module::Builder device_module_builder;
};

}  // namespace detail

}  // namespace backends
}  // namespace cinn

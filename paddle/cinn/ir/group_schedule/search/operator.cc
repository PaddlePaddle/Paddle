// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/search/operator.h"
#include <sstream>

#include "glog/logging.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

COMMON_DECLARE_bool(print_ir);
PD_DECLARE_string(cinn_kernel_execution_label);

namespace cinn {
namespace ir {
namespace search {

static std::shared_ptr<pir::PassManager> CreatePassManager() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  auto pass_manager = std::make_shared<pir::PassManager>(ctx);
  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }
  return pass_manager;
}

Operator::Operator(::pir::Program* program) : main_program_(program) {
  std::stringstream ss;
  ss << *main_program_;
}

Operator::Operator(::pir::Program* main_program,
                   ::pir::Program* startup_program)
    : main_program_(main_program), startup_program_(startup_program) {
  std::stringstream ss;
  ss << *main_program;
}

void Operator::Prepare() {
  VLOG(4) << "[Debug] ============= Operator::Prepare() Begin ============= \n";
  if (startup_program_ == nullptr) {
    VLOG(4) << "[Debug] Operator::Prepare() Skip due to startup_program_ == "
               "nullptr";
    return;
  }
  ::pir::IrMapping ir_mapping;
  VLOG(4) << "[Debug] Operator::Prepare() Before Clone ir_mapping";
  std::shared_ptr<::pir::Program> program_cloned =
      startup_program_->Clone(ir_mapping);
  VLOG(4) << "[Debug] Operator::Prepare() Before ApplyCinnPass";
  cinn::dialect::ir::ApplyCinnPass(program_cloned.get(), CreatePassManager);
  VLOG(4) << "[Debug] Operator::Prepare() Before PdOpLowerToKernelPass";
  kernel_program_ = std::move(
      paddle::dialect::PdOpLowerToKernelPass(program_cloned.get(), place_));
  VLOG(4) << "[Debug] Operator::Prepare() Before executor_.reset";
  std::vector<std::string> fetch_var_names{};
  executor_.reset(new paddle::framework::InterpreterCore(
      place_, fetch_var_names, kernel_program_->block(), exe_scope_.get()));
  VLOG(4) << "[Debug] Operator::Prepare() Before executor_.Run";
  std::vector<std::string> feed_names{};
  executor_->Run(feed_names, true);
  VLOG(4) << "[Debug] ============= Operator::Prepare() End ============= \n";
}

void Operator::Compile() {
  Prepare();
  // auto w0 = exe_scope_->FindVar("conv2d_0.w_0")->Get<phi::DenseTensor>();
  // const float* w0_cpu = w0.data<float>();
  // std::stringstream ss;
  // ss << "w0 = [";
  // for (size_t i = 0; i < w0.numel(); ++i) {
  //   ss << w0_cpu[i] << ", ";
  // }
  // ss << " ]";
  // VLOG(6) << ss.str();

  ::pir::IrMapping ir_mapping;
  VLOG(4) << "[Debug] Operator::Compile() Before Clone ir_mapping";
  std::shared_ptr<::pir::Program> program_cloned =
      main_program_->Clone(ir_mapping);
  VLOG(4) << "[Debug] Operator::Compile() Before ApplyCinnPass";
  cinn::dialect::ir::ApplyCinnPass(program_cloned.get(), CreatePassManager);
  VLOG(4) << "[Debug] Operator::Compile() Before PdOpLowerToKernelPass";
  kernel_program_ = std::move(
      paddle::dialect::PdOpLowerToKernelPass(program_cloned.get(), place_));
  VLOG(4) << "[Debug] Operator::Compile() Before executor_.reset";
  executor_.reset(new paddle::framework::InterpreterCore(
      place_, {"out@fetch"}, kernel_program_->block(), exe_scope_.get()));
  VLOG(4) << "[Debug] Operator::Compile() Kernel after Operator compile: \n"
          << *kernel_program_;
}

phi::DenseTensor Operator::Run(
    const std::unordered_map<std::string, std::shared_ptr<phi::DenseTensor>>&
        input_name_and_tensor) {
  std::vector<std::string> input_names;
  std::vector<phi::DenseTensor> input_tensors;
  for (const auto& item : input_name_and_tensor) {
    // LOG(INFO) << "input_name: " << item.first;

    // for (int i = 0; i < item.second.size(); ++i) {
    //   LOG(INFO) << "dim[" << i << "]: " << item.second[i];
    // }
    input_names.push_back(item.first);
    input_tensors.push_back(*item.second);
  }

  auto fetch_list = executor_->Run(input_names, input_tensors, true);
  // for (size_t i = 0; i < fetch_list.size(); ++i) {
  //   const float* fetch_data =
  //   PADDLE_GET_CONST(phi::DenseTensor, fetch_list[i]).data<float>();
  //   VLOG(7) << "fetch_data[" << i << "] =  " << *fetch_data;
  // }
  auto tensor = PADDLE_GET_CONST(phi::DenseTensor, fetch_list[0]);
  VLOG(3) << "[Debug] Output Tensor: "
          << paddle::framework::PrintDenseTensor(&tensor, 0, 50);
  return tensor;
}

}  // namespace search
}  // namespace ir
}  // namespace cinn

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

#include "paddle/fluid/sub_graph/sub_graph_checker.h"

#include <chrono>
#include <ctime>
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/include/core/ir_context.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_cluster_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/lowering_pass/lower_cinn_fusion_op_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/framework/new_executor/interpretercore.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/pir/include/core/operation_utils.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

namespace paddle {
namespace test {

bool AllClose(const phi::DenseTensor& a,
              const phi::DenseTensor& b,
              const float rtol = 1e-5,
              const float atol = 1e-8) {
  if (a.dims() != b.dims() || a.dtype() != b.dtype()) {
    return false;
  }

  if (a.dtype() == phi::DataType::FLOAT32) {
    auto pa = a.data<float>();
    auto pb = b.data<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
      if (std::abs(pa[i] - pb[i]) > (atol + rtol * std::abs(pb[i]))) {
        LOG(WARNING) << "element pos " << i << "\t" << pa[i] << "\t" << pb[i]
                     << std::endl;
        return false;
      }
    }
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "ONLY support float32, but received %s", a.dtype()));
  }

  return true;
}

std::vector<pir::Value> GetBlockInput(pir::Block* block) {
  std::vector<pir::Value> inputs;
  std::unordered_set<::pir::Value> inner_output;
  for (auto& op : *block) {
    for (size_t i = 0; i < op.num_results(); ++i) {
      inner_output.insert(op.result(i));
    }

    if (op.isa<paddle::dialect::DataOp>()) {
      inputs.push_back(op.result(0));
    }
  }

  std::unordered_set<::pir::Value> value_set;
  for (auto& op : *block) {
    for (size_t i = 0; i < op.num_operands(); ++i) {
      if (!op.operand(i) || !(op.operand_source(i))) {
        continue;
      }
      auto value = op.operand_source(i);
      if (!inner_output.count(value) && !value_set.count(value)) {
        inputs.push_back(value);
        value_set.insert(value);
      }
    }
  }
  return inputs;
}

SubGraphChecker::SubGraphChecker(std::shared_ptr<pir::Program> phi_program,
                                 std::shared_ptr<pir::Program> prim_program)
    : phi_program_(phi_program), prim_program_(prim_program) {}

bool SubGraphChecker::CheckResult() {
  auto phi_res = RunPhiResult();
  auto cinn_res = RunCinnResult();

  bool check_right = true;
  for (size_t i = 0; i < phi_res.size(); ++i) {
    auto res = AllClose(phi_res[i], cinn_res[i]);
    if (!res) {
      check_right = false;
      break;
    }
    VLOG(3) << "compare index " << i << "\t" << res << std::endl;
  }

  if (check_right) {
    LOG(INFO) << "Result check Success" << std::endl;
  } else {
    LOG(INFO) << "Result check Failed" << std::endl;
  }
  return check_right;
}

std::vector<phi::DenseTensor> SubGraphChecker::RunPhiResult() {
  phi_input_values_ = GetBlockInput(phi_program_->block());
  InitInputs(phi_input_values_, phi_program_->block(), &inner_scope_);
  AppendFetchOp(phi_program_->block(), &phi_fetch_names_, kOutputPrefix);

  phi::Place place = phi::GPUPlace(0);
  phi_kernel_program_ =
      paddle::dialect::PdOpLowerToKernelPass(phi_program_.get(), place);

  paddle::framework::interpreter::ExecutionConfig exec_config;
  exec_config.create_local_scope = false;
  for (size_t i = 0; i < phi_input_values_.size(); ++i) {
    std::string name = kInputPrefix + std::to_string(i);
    exec_config.skip_gc_vars.insert(name);
  }

  std::vector<std::string> fetch_var_names;
  for (auto name : phi_fetch_names_) {
    fetch_var_names.push_back(name + kFetchSuffix);
  }
  paddle::framework::InterpreterCore exec(place,
                                          fetch_var_names,
                                          phi_kernel_program_->block(),
                                          &inner_scope_,
                                          exec_config);

  exec.Run({}, true);

  std::vector<phi::DenseTensor> vec_res;
  for (auto& name : fetch_var_names) {
    vec_res.push_back(inner_scope_.FindVar(name)->Get<phi::DenseTensor>());
  }

  return vec_res;
}

std::vector<phi::DenseTensor> SubGraphChecker::RunCinnResult() {
  cinn_input_values_ = GetBlockInput(prim_program_->block());

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();

  AppendFetchOp(prim_program_->block(), &cinn_fetch_names_, kOutputPrefix);

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
  pm.AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pm.Run(prim_program_.get());

  phi::Place place = phi::GPUPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(prim_program_.get(), place);

  std::vector<std::string> fetch_var_names;
  for (auto name : cinn_fetch_names_) {
    fetch_var_names.push_back(name + kFetchSuffix);
  }

  paddle::framework::interpreter::ExecutionConfig exec_config;
  exec_config.create_local_scope = false;
  for (size_t i = 0; i < phi_input_values_.size(); ++i) {
    std::string name = kInputPrefix + std::to_string(i);
    exec_config.skip_gc_vars.insert(name);
  }

  paddle::framework::InterpreterCore executor(place,
                                              fetch_var_names,
                                              kernel_program->block(),
                                              &inner_scope_,
                                              exec_config);

  executor.Run({}, true);

  std::vector<phi::DenseTensor> vec_res;
  for (auto& name : fetch_var_names) {
    vec_res.push_back(inner_scope_.FindVar(name)->Get<phi::DenseTensor>());
  }

  return vec_res;
}

std::vector<double> SubGraphChecker::CheckSpeed() {
  auto time_phi = RunPhiSpeed();
  auto time_cinn = RunCinnSpeed();

  LOG(INFO) << "time cost: Phi: " << time_phi << "\tCINN : " << time_cinn
            << std::endl;

  std::vector<double> speed_data{time_phi, time_cinn};
  return speed_data;
}

double SubGraphChecker::RunPhiSpeed() {
  RemoveFetchOp(phi_program_->block());
  phi::Place place = phi::GPUPlace(0);
  phi_kernel_program_ =
      paddle::dialect::PdOpLowerToKernelPass(phi_program_.get(), place);

  paddle::framework::interpreter::ExecutionConfig exec_config;
  exec_config.create_local_scope = false;
  for (size_t i = 0; i < phi_input_values_.size(); ++i) {
    std::string name = kInputPrefix + std::to_string(i);
    exec_config.skip_gc_vars.insert(name);
  }

  std::vector<std::string> fetch_var_names;
  for (auto name : phi_fetch_names_) {
    fetch_var_names.push_back(name + kFetchSuffix);
  }
  paddle::framework::InterpreterCore exec(place,
                                          fetch_var_names,
                                          phi_kernel_program_->block(),
                                          &inner_scope_,
                                          exec_config);
  // warm up
  for (size_t i = 0; i < 10; ++i) {
    exec.Run({}, true);
  }

  auto start = std::chrono::system_clock::now();
  for (size_t i = 0; i < 10000; ++i) {
    exec.Run({}, true);
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;

  return elapsed_seconds.count();
}
double SubGraphChecker::RunCinnSpeed() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();

  AppendFetchOp(prim_program_->block(), &cinn_fetch_names_, kOutputPrefix);

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  pir::PassManager pm(ctx);
  pm.AddPass(cinn::dialect::ir::CreatePdOpToCinnOpPass());
  pm.AddPass(cinn::dialect::ir::CreateAddBroadcastToElementwisePass());
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.AddPass(cinn::dialect::ir::CreateCinnGroupClusterPass());
  pm.AddPass(cinn::dialect::ir::CreateLowerCinnFusionOpPass());
  pm.Run(prim_program_.get());

  phi::Place place = phi::GPUPlace(0);

  RemoveFetchOp(prim_program_->block());

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(prim_program_.get(), place);

  std::vector<std::string> fetch_var_names;
  for (auto name : cinn_fetch_names_) {
    fetch_var_names.push_back(name + kFetchSuffix);
  }

  paddle::framework::interpreter::ExecutionConfig exec_config;
  exec_config.create_local_scope = false;
  for (size_t i = 0; i < phi_input_values_.size(); ++i) {
    std::string name = kInputPrefix + std::to_string(i);
    exec_config.skip_gc_vars.insert(name);
  }

  paddle::framework::InterpreterCore executor(place,
                                              fetch_var_names,
                                              kernel_program->block(),
                                              &inner_scope_,
                                              exec_config);

  for (size_t i = 0; i < 100; ++i) {
    executor.Run({}, true);
  }

  auto start = std::chrono::system_clock::now();
  for (size_t i = 0; i < 10000; ++i) {
    executor.Run({}, true);
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;

  return elapsed_seconds.count();
}

void SubGraphChecker::RemoveFetchOp(pir::Block* block) {
  for (auto it = block->begin(); it != block->end();) {
    if (it->isa<paddle::dialect::FetchOp>()) {
      it = block->erase(it);
    } else {
      it++;
    }
  }
}

void SubGraphChecker::InitInputs(const std::vector<pir::Value>& input_values,
                                 pir::Block* block,
                                 paddle::framework::Scope* scope) {
  // build a program, init data and set parameter to scope
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  phi::DenseTensor* out_tensor;
  for (size_t i = 0; i < input_values.size(); ++i) {
    auto tensor_type =
        input_values[i].type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto shape = common::vectorize<int64_t>(tensor_type.dims());
    auto random =
        builder
            .Build<paddle::dialect::UniformOp>(
                shape,
                paddle::dialect::TransToPhiDataType(tensor_type.dtype()),
                -0.2,
                0.2,
                0,
                phi::GPUPlace())
            .result(0);
    auto name = kInputPrefix + std::to_string(i);
    builder.Build<pir::SetParameterOp>(random, name);
    auto param = scope->Var(name);
    out_tensor = param->GetMutable<phi::DenseTensor>();
  }

  if (input_values.size() > 0) {
    phi::Place place = phi::GPUPlace(0);

    auto kernel_program =
        paddle::dialect::PdOpLowerToKernelPass(program.get(), place);

    paddle::framework::interpreter::ExecutionConfig exec_config;
    exec_config.create_local_scope = false;
    paddle::framework::InterpreterCore executor(
        place, {}, kernel_program->block(), scope, exec_config);

    executor.Run({}, true);
  }
}
void SubGraphChecker::AppendGetParameter(
    const std::vector<pir::Value>& input_values, pir::Block* block) {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  ::pir::Builder builder = ::pir::Builder(ctx, block);
  builder.SetInsertionPointToStart(block);
  for (size_t i = 0; i < input_values.size(); ++i) {
    auto get_param =
        builder
            .Build<pir::ParameterOp>(kInputPrefix + std::to_string(i),
                                     input_values[i].type())
            .result(0);

    for (auto it = input_values[i].use_begin();
         it != input_values[i].use_end();) {
      (it++)->set_source(get_param);
    }
  }
}

void SubGraphChecker::AppendFetchOp(pir::Block* block,
                                    std::vector<std::string>* fetch_names,
                                    const std::string& prefix) {
  for (auto& op : *block) {
    if (op.isa<paddle::dialect::FetchOp>()) {
      fetch_names->push_back(
          op.attribute("name").dyn_cast<pir::StrAttribute>().AsString());
    }
  }

  if (fetch_names->size() > 0) {
    return;
  }

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  ::pir::Builder builder = ::pir::Builder(ctx, block);

  for (size_t i = 0; i < block->back().num_results(); ++i) {
    auto name = prefix + std::to_string(i);
    builder.Build<paddle::dialect::FetchOp>(block->back().result(i), name, i);

    fetch_names->push_back(name);
  }
}

}  // namespace test
}  // namespace paddle

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
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/pir/core/ir_context.h"

#include "paddle/cinn/hlir/dialect/operator/ir/manual_op.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/cinn_group_lowering_pass.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/pd_to_cinn_pass.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/pir/core/operation_utils.h"
#include "paddle/pir/pass/pass.h"
#include "paddle/pir/pass/pass_manager.h"

namespace paddle {
namespace test {

bool AllClose(const phi::DenseTensor& a,
              const phi::DenseTensor& b,
              float rtol = 1e-5,
              float atol = 1e-8) {
  if (a.dims() != b.dims()) {
    return false;
  }

  if (a.dtype() != b.dtype()) {
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
    PADDLE_THROW(phi::errors::Unimplemented("ONLY support float32 "));
  }

  return true;
}

pir::Operation* BuildOpFrom(
    pir::Operation* to_copy_op,
    std::unordered_map<pir::Value, pir::Value>& value_map) {  // NOLINT
  pir::OperationArgument to_create_argument(to_copy_op->info());
  to_create_argument.attributes = to_copy_op->attributes();

  VLOG(6) << "start copy op: " << to_copy_op->name();
  auto origin_results = to_copy_op->results();
  VLOG(6) << "start translate origin results into op type.";
  std::transform(origin_results.begin(),
                 origin_results.end(),
                 std::back_inserter(to_create_argument.output_types),
                 [](const pir::OpResult& r) {
                   // OpResult -> OpType
                   return r.type();
                 });

  // transform by value_map dict.
  VLOG(6) << "start create op.";
  auto origin_operands = to_copy_op->operands();
  std::transform(origin_operands.begin(),
                 origin_operands.end(),
                 std::back_inserter(to_create_argument.inputs),
                 [&value_map](const pir::OpOperand& operand) {
                   // Operand -> OpResult
                   return value_map[operand.source()];
                 });
  auto* cloned_op = pir::Operation::Create(std::move(to_create_argument));

  std::vector<int> tmp;
  std::transform(
      origin_results.begin(),
      origin_results.end(),
      cloned_op->results().begin(),
      std::back_inserter(tmp),  // NOLINT, just a placeholder.
      [&value_map](const pir::OpResult& a, const pir::OpResult& b) {  // NOLINT
        value_map[a.Value::impl()] = b.Value::impl();
        return 1;
      });
  return cloned_op;
}

using OpResultMap = std::unordered_map<pir::OpResult, pir::OpResult>;
std::shared_ptr<pir::Program> CloneProgram(const pir::Program& program) {
  // Limitation of this function:
  // 1. don't support Parameters.
  // 2. don't support Regions in operator.
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto cloned_program = std::make_shared<pir::Program>(ctx);
  std::unordered_map<pir::Value, pir::Value> value_map;
  for (auto& op : *program.block()) {
    auto* cloned_op = BuildOpFrom(op, value_map);
    cloned_program->block()->push_back(cloned_op);
  }
  std::unordered_map<pir::OpResult, pir::OpResult> op_result_map;
  for (auto& pair : value_map) {
    op_result_map[pair.first.dyn_cast<pir::OpResult>()] =
        pair.second.dyn_cast<pir::OpResult>();
  }
  return cloned_program;
}

std::vector<pir::Value> GetBlockInput(pir::Block* block) {
  std::vector<pir::Value> vec_res;
  std::unordered_set<::pir::Value> block_inner_output;
  for (auto op : *block) {
    for (size_t i = 0; i < op->num_results(); ++i) {
      block_inner_output.insert(op->result(i));
    }

    if (op->isa<paddle::dialect::DataOp>()) {
      vec_res.push_back(op->result(0));
    }
  }

  std::unordered_set<::pir::Value> insert_value;
  for (auto op : *block) {
    for (size_t i = 0; i < op->num_operands(); ++i) {
      if (!block_inner_output.count(op->operand_source(i)) &&
          !insert_value.count(op->operand_source(i))) {
        vec_res.push_back(op->operand_source(i));
        insert_value.insert(op->operand_source(i));
      }
    }
  }
  return vec_res;
}

SubGraphChecker::SubGraphChecker(std::shared_ptr<pir::Program> phi_program,
                                 std::shared_ptr<pir::Program> prim_program)
    : phi_program_(CloneProgram(*(phi_program.get()))),
      prim_program_(CloneProgram(*(prim_program.get())))
// : phi_program_(phi_program), prim_program_( prim_program)
{}

void SubGraphChecker::CheckResult1() {
  std::cerr << "~~~\n";
  auto phi_res = RunPhiResult();

  std::cerr << "@@@@\n";

  std::cerr << "finish phi run\n";

  auto cinn_res = RunCinnResult();

  for (size_t i = 0; i < phi_res.size(); ++i) {
    auto res = AllClose(phi_res[i], cinn_res[i]);

    std::cerr << "compare " << i << "\t" << res << std::endl;
  }
}

std::vector<phi::DenseTensor> SubGraphChecker::RunPhiResult() {
  phi_input_values_ = GetBlockInput(phi_program_->block());
  InitInputs(phi_input_values_, phi_program_->block(), &inner_scope_);
  std::cerr << "--11\n";
  phi_program_->Print(std::cout);
  // AppendGetParameter( phi_input_values_, phi_program_->block() );

  std::cerr << "12\n";
  phi_program_->Print(std::cout);

  std::cerr << "13\n";
  AppendFetchOp(phi_program_->block(), &phi_fetch_names_, "phi_out_");

  phi_program_->Print(std::cout);

  std::cerr << "15\n";
  paddle::platform::Place place = paddle::platform::CUDAPlace(0);
  phi_kernel_program_ =
      paddle::dialect::PdOpLowerToKernelPass(phi_program_.get(), place);

  phi_kernel_program_->Print(std::cout);

  std::cerr << "init kernel program\n";

  paddle::framework::interpreter::ExecutionConfig exec_config;
  exec_config.create_local_scope = false;
  exec_config.skip_gc_vars.insert("input_0");

  std::vector<std::string> fetch_var_names;
  for (auto name : phi_fetch_names_) {
    fetch_var_names.push_back(name + "@fetch");
  }
  phi_exec_ =
      new paddle::framework::InterpreterCore(place,
                                             fetch_var_names,
                                             phi_kernel_program_->block(),
                                             &inner_scope_,
                                             exec_config);

  phi_exec_->Run({}, true);

  std::cerr << "finish phi kernel run\n";

  std::vector<phi::DenseTensor> vec_res;
  for (auto& name : fetch_var_names) {
    vec_res.push_back(
        inner_scope_.FindVar("phi_out_0@fetch")->Get<phi::DenseTensor>());
  }

  return vec_res;
}

std::vector<phi::DenseTensor> SubGraphChecker::RunCinnResult() {
  std::cerr << "begin cinn\n";
  cinn_input_values_ = GetBlockInput(prim_program_->block());
  // InitInputs( cinn_input_values_, prim_program_->block(), &inner_scope_);

  std::cerr << "finish init\n";
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();

  prim_program_->Print(std::cout);
  // AppendGetParameter(cinn_input_values_, prim_program_->block() );
  AppendFetchOp(prim_program_->block(), &cinn_fetch_names_, "cinn_out_");

  std::cerr << "11\n";
  prim_program_->Print(std::cout);

  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();

  cinn::dialect::ir::PdOp2CinnOpConverter(prim_program_.get());

  std::cerr << "12\n";
  prim_program_->Print(std::cout);

  pir::PassManager pm(ctx);
  pm.AddPass(
      std::make_unique<cinn::dialect::ir::AddBroadcastToElementwisePass>());
  pm.AddPass(pir::CreateBuildCinnPass());
  pm.Run(prim_program_.get());

  std::cerr << "13\n";
  prim_program_->Print(std::cout);

  auto res = cinn::dialect::ir::CINNGroupLoweringPass(prim_program_.get());

  std::cerr << "15\n";
  res->Print(std::cout);
  paddle::platform::Place place = paddle::platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(res.get(), place);

  std::vector<std::string> fetch_var_names;
  for (auto name : cinn_fetch_names_) {
    fetch_var_names.push_back(name + "@fetch");
  }
  paddle::framework::InterpreterCore executor(
      place, fetch_var_names, kernel_program->block(), &inner_scope_);

  executor.Run({}, true);

  std::vector<phi::DenseTensor> vec_res;
  for (auto& name : fetch_var_names) {
    vec_res.push_back(inner_scope_.FindVar(name)->Get<phi::DenseTensor>());
  }

  return vec_res;
}

void SubGraphChecker::InitInputs(const std::vector<pir::Value>& input_values,
                                 pir::Block* block,
                                 paddle::framework::Scope* scope) {
  std::cerr << "inpuut value szi " << input_values.size() << std::endl;

  // build a proram, init data and set parameter to scope
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  auto program = std::make_shared<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  phi::DenseTensor* out_tensor;
  for (size_t i = 0; i < input_values.size(); ++i) {
    auto tensor_type =
        input_values[i].type().dyn_cast<paddle::dialect::DenseTensorType>();
    auto shape = phi::vectorize<int64_t>(tensor_type.dims());
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
    auto name = "input_" + std::to_string(i);
    builder.Build<pir::SetParameterOp>(random, name);
    auto param = scope->Var(name);
    out_tensor = param->GetMutable<phi::DenseTensor>();

    std::cerr << "out ptr" << out_tensor << std::endl;
  }

  if (input_values.size() > 0) {
    paddle::platform::Place place = paddle::platform::CUDAPlace(0);

    program->Print(std::cout);
    auto kernel_program =
        paddle::dialect::PdOpLowerToKernelPass(program.get(), place);

    std::cerr << "init kernel program\n";
    kernel_program->Print(std::cout);
    paddle::framework::interpreter::ExecutionConfig exec_config;
    exec_config.create_local_scope = false;
    paddle::framework::InterpreterCore executor(
        place, {}, kernel_program->block(), scope, exec_config);

    executor.Run({}, true);
    std::cerr << "out ptr" << out_tensor->data() << std::endl;
    std::cerr << "after init " << *out_tensor << std::endl;
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
            .Build<pir::GetParameterOp>("input_" + std::to_string(i),
                                        input_values[i].type())
            .result(0);

    std::cerr << "use count " << input_values[i].use_count() << std::endl;
    for (auto it = input_values[i].use_begin();
         it != input_values[i].use_end();) {
      std::cerr << "set sorce \n";
      (it++)->set_source(get_param);
    }
  }
}

void SubGraphChecker::AppendFetchOp(pir::Block* block,
                                    std::vector<std::string> fetch_names,
                                    const std::string& prefix) {
  for (auto op : *block) {
    if (op->isa<paddle::dialect::FetchOp>()) {
      fetch_names->push_back(
          op->attribute("name").dyn_cast<pir::StrAttribute>().AsString());
    }
  }

  if (fetch_names->size() > 0) {
    return;
  }

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

  ::pir::Builder builder = ::pir::Builder(ctx, block);

  std::cerr << "!!!!!!!!!!!!! " << block->back()->name() << std::endl;
  for (size_t i = 0; i < block->back()->num_results(); ++i) {
    auto name = prefix + std::to_string(i);
    builder.Build<paddle::dialect::FetchOp>(block->back()->result(i), name, i);

    fetch_names->push_back(name);
  }
}

}  // namespace test
}  // namespace paddle

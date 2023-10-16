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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "paddle/fluid/framework/new_executor/interpretercore.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"

#include "paddle/cinn/hlir/dialect/operator/ir/op_attribute.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/new_ir_compiler.h"
#include "paddle/cinn/utils/data_util.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/backends/gpu/gpu_context.h"

std::unique_ptr<::pir::Program> BuildProgram() {
  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  auto program = std::make_unique<::pir::Program>(ctx);
  ::pir::Builder builder = ::pir::Builder(ctx, program->block());

  const float value = 0.5;
  auto full_op_x =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto full_op_y =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());
  auto full_op_z =
      builder.Build<paddle::dialect::FullOp>(std::vector<int64_t>{2, 2},
                                             value,
                                             phi::DataType::FLOAT32,
                                             phi::GPUPlace());

  auto sin = builder.Build<paddle::dialect::SinOp>(full_op_x.result(0));
  auto cos = builder.Build<paddle::dialect::CosOp>(full_op_y.result(0));
  auto add =
      builder.Build<paddle::dialect::AddOp>(sin.result(0), cos.result(0));
  builder.Build<paddle::dialect::FetchOp>(add.out(), "out", 0);
  return std::move(program);
}

namespace paddle {
namespace framework {

TEST(CinnJitInstruction, Run) {
  // Step 1: Construct pir::Program
  std::unique_ptr<::pir::Program> program = BuildProgram();
  // EXPECT_EQ(program->block()->size(), 2u);

  // Step 2: Compiler New pir::Program into Runtime Program
  auto target = cinn::common::DefaultNVGPUTarget();
  auto scope = cinn::hlir::framework::BuildScope(target, *program);
  // ASSERT_EQ(scope->var_names().size(), 2);

  program->Print(std::cout);

  std::vector<cinn::hlir::framework::NewIRCompiler*> compiler_list;

  // std::set<std::string> using_cinn_ops = {"pd_op.full", "pd_op.sin",
  // "pd_op.cos", "pd_op.add"};
  std::set<std::string> using_cinn_ops = {"pd_op.sin", "pd_op.cos"};
  // std::vector<cinn::hlir::framework::newir::GroupPtr> groups;
  // for (auto it = program->block()->begin(); it != program->block()->end();
  //      ++it) {
  //   std::vector<::pir::Operation*> ops = {*it};
  //   groups.push_back(
  //       );
  // }

  // auto fn_ptr_res = ir_compiler.BuildCUDAJITInfo(groups);

  ::pir::IrContext* ctx = ::pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<cinn::dialect::RuntimeDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<paddle::dialect::KernelDialect>();
  auto ir_program = std::make_unique<::pir::Program>(ctx);
  std::string jit_op_name = cinn::dialect::JitKernelOp::name();
  ::pir::OpInfo op_info = ctx->GetRegisteredOpInfo(jit_op_name);

  int index = 0;
  std::unordered_map<pir::Value, pir::Value> value_map;
  for (auto it = program->block()->begin(); it != program->block()->end();
       ++it) {
    if (using_cinn_ops.count((*it)->name())) {
      auto ir_compiler =
          new cinn::hlir::framework::NewIRCompiler(*program, target, scope);

      std::vector<::pir::Operation*> ops = {*it};
      auto group = std::make_shared<cinn::hlir::framework::newir::Group>(ops);
      auto fn_ptr_res = ir_compiler->BuildCUDAJITInfo({group});
      compiler_list.push_back(ir_compiler);
      std::cerr << "build attr" << std::endl;
      std::unordered_map<std::string, ::pir::Attribute> op_attrs{
          {cinn::dialect::JitKernelOp::kAttrName,
           cinn::dialect::CUDAJITInfoAttribute::get(ctx, fn_ptr_res[0])},
      };

      std::cerr << "build fin" << std::endl;
      auto type1 = (*it)->result(0).type();
      if (!type1) {
        std::cerr << "wroing with type1" << std::endl;
      }

      std::vector<pir::Value> vec_ins;

      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        vec_ins.push_back(value_map.at((*it)->operand_source(i)));
      }

      std::cerr << "fin get allocated dense tensor" << std::endl;
      ::pir::Operation* cinn_op =
          ::pir::Operation::Create(vec_ins, op_attrs, {type1}, op_info);

      value_map[(*it)->result(0)] = cinn_op->result(0);

      ir_program->block()->push_back(cinn_op);

    } else {
      std::vector<pir::Value> vec_ins;

      for (size_t i = 0; i < (*it)->num_operands(); ++i) {
        vec_ins.push_back(value_map.at((*it)->operand_source(i)));
      }

      auto type1 = (*it)->result(0).type();
      ::pir::OpInfo info1 = ctx->GetRegisteredOpInfo((*it)->name());
      ::pir::Operation* op = ::pir::Operation::Create(
          vec_ins, (*it)->attributes(), {type1}, info1);

      ir_program->block()->push_back(op);

      value_map[(*it)->result(0)] = op->result(0);
    }
  }

  //
  // for (auto& var_name : scope->var_names()) {
  //   std::string name = {var_name.begin(), var_name.end()};
  //   out_names.insert(name);
  // }

  ir_program->Print(std::cout);
  platform::Place place = platform::CUDAPlace(0);

  auto kernel_program =
      paddle::dialect::PdOpLowerToKernelPass(ir_program.get(), place);

  Scope exe_scope;

  kernel_program->Print(std::cout);
  paddle::framework::interpreter::ExecutionConfig exe_conf;
  exe_conf.create_local_scope = false;
  InterpreterCore executor(
      place, {"out@fetch"}, kernel_program->block(), &exe_scope);

  std::set<std::string> out_names;
  out_names.insert("out@fetch");
  auto local_names = exe_scope.LocalVarNames();
  for (size_t i = 0; i < local_names.size(); ++i) {
    out_names.insert(local_names[i]);
    std::cerr << "out name s " << local_names[i] << std::endl;
  }

  executor.SetSkipGcVars(out_names);
  auto out = executor.Run({}, true);

  std::cout << out.size() << std::endl;
  std::cerr << "exe scope "
            << (executor.local_scope()->FindVar("out@fetch") != nullptr)
            << std::endl;
  std::cerr
      << executor.local_scope()->FindVar("out@fetch")->Get<phi::DenseTensor>()
      << std::endl;
  // std::cerr << "out " << PADDLE_GET_CONST(phi::DenseTensor, out[0]) <<
  // std::endl;

  // auto dev_ctx = phi::DeviceContextPool::Instance().Get(place);
  // auto gpu_ctx = static_cast<phi::GPUContext*>(dev_ctx);

  // auto in_tesnor =
  // exe_scope.Var("0x5bfff50_inner_var_0")->Get<phi::DenseTensor>(); std::cerr
  // << "out " << in_tesnor << std::endl;

  // phi::DenseTensor cpu_tensor;
  // phi::Copy(*gpu_ctx, in_tesnor, phi::CPUPlace(), true, d_in3.get());

  // // TODO(Aurelius84): Need to replace check with framework::Scope.
  // const float value = 2.0;
  // for (auto& name : out_names) {
  //   std::vector<float> data =
  //       cinn::GetTensorData<float>(scope->GetTensor(name), target);
  //   for (int i = 0; i < data.size(); ++i) {
  //     LOG_FIRST_N(INFO, 3) << "data: " << data[i];
  //     ASSERT_NEAR(data[i], value, 1e-5);
  //   }
  // }
}

}  // namespace framework
}  // namespace paddle

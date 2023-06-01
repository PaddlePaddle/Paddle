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

#include <gtest/gtest.h>

#include "paddle/fluid/dialect/pd_dialect.h"
#include "paddle/fluid/dialect/pd_interface.h"
#include "paddle/fluid/dialect/pd_op.h"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/utils.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/dialect/pd_attribute.h"

void build_scope(ir::Block* block,
                 paddle::framework::Scope* scope,
                 std::unordered_map<void*, std::string>* name_map) {
  std::unordered_map<ir::Value, int> map_test;

  int count = 0;
  for (auto it = block->begin(); it != block->end(); ++it) {
    std::cerr << (*it)->op_name() << std::endl;

    int input = (*it)->num_operands();
    if (input > 0) {
      for (int i = 0; i < input; ++i) {
        std::cerr << "input "
                  << (*it)->GetOperandByIndex(i).impl()->source().impl()
                  << std::endl;

        void* ptr = (*it)->GetOperandByIndex(i).impl()->source().impl();
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
        }
        auto var = scope->Var(name);
        // need to update here, only support DenseTensor
        var->GetMutable<phi::DenseTensor>();
      }
    }

    int out_num = (*it)->num_results();

    if (out_num > 0) {
      for (int i = 0; i < out_num; ++i) {
        std::cerr << "output " << (*it)->GetResultByIndex(i).impl()
                  << std::endl;
        // map_test[ (*it)->GetResultByIndex(i) ] = 1;
        void* ptr = (*it)->GetResultByIndex(i).impl();
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
        }
        auto var = scope->Var(name);

        var->GetMutable<phi::DenseTensor>();
      }
    }
  }
}

TEST(program_test, program) {
  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

  ir::Program program;
  ir::Type fp32_dtype = ir::Float32Type::get(ctx);

  paddle::dialect::DenseTensorTypeStorage::Dim dims = {2, 2};
  paddle::dialect::DenseTensorTypeStorage::DataLayout data_layout =
      paddle::dialect::DenseTensorTypeStorage::DataLayout::NCHW;
  paddle::dialect::DenseTensorTypeStorage::LoD lod = {};
  size_t offset = 0;
  ir::Type dense_tensor_dtype = paddle::dialect::DenseTensorType::get(
      ctx, fp32_dtype, dims, data_layout, lod, offset);

  // (1) Def a = GetParameterOp("a")
  std::string op1_name = std::string(paddle::dialect::UniformOp::name());
  ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op1_name);
  ir::Attribute ten = ir::Int32_tAttribute::get(ctx, 2);

  // ir::Attribute shape_1 = ir::ArrayAttribute::get(ctx, {ten} );
  std::unordered_map<std::string, ir::Attribute> op1_attribute{{"shape", ten}};
  std::cerr << "ten " << ten.dyn_cast<ir::Int32_tAttribute>().data()
            << std::endl;
  ir::Operation* op1 =
      ir::Operation::create({}, op1_attribute, {dense_tensor_dtype}, op1_info);

  program.InsertOp(op1);

  // (2) Def b = GetParameterOp("b")
  std::string op2_name = std::string(paddle::dialect::UniformOp::name());
  ir::OpInfo op2_info = ctx->GetRegisteredOpInfo(op2_name);
  ir::Attribute ten2 = ir::Int32_tAttribute::get(ctx, 3);
  std::unordered_map<std::string, ir::Attribute> op2_attribute{{"shape", ten2}};
  ir::Operation* op2 =
      ir::Operation::create({}, op2_attribute, {dense_tensor_dtype}, op2_info);
  program.InsertOp(op2);

  // (3) Def out = AddOp(a, b)
  std::string add_op_name = std::string(paddle::dialect::AddOp::name());
  ir::OpInfo add_op_info = ctx->GetRegisteredOpInfo(add_op_name);
  ir::Type output_type = ir::Float32Type::get(ctx);
  ir::Operation* add_op = ir::Operation::create(
      {op1->GetResultByIndex(0), op2->GetResultByIndex(0)},
      {},
      {output_type},
      add_op_info);
  program.InsertOp(add_op);

  std::cerr << "begin" << std::endl;
  std::cerr << program << std::endl;

  std::cerr << "fin" << std::endl;

  auto block = program.block();
  std::unordered_map<void*, std::string> name_map;

  paddle::framework::Scope scope;

  build_scope(program.block(), &scope, &name_map);

  for (auto it = name_map.begin(); it != name_map.end(); ++it) {
    std::cerr << it->first << "\t" << it->second << std::endl;
  }

  phi::Place cpu_place(phi::AllocationType::CPU);
  for (auto it = block->begin(); it != block->end(); ++it) {
    std::cerr << (*it)->op_name() << std::endl;

    auto attr_map = (*it)->attribute();
    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());

    for (auto it1 = attr_map.begin(); it1 != attr_map.end(); it1++) {
      std::cerr << it1->first << std::endl;
      std::cerr << it1->second.dyn_cast<ir::Int32_tAttribute>().data()
                << std::endl;
      // auto t1 = it1->second.dyn_cast<ir::ArrayAttribute>();
      // std::cerr << t1[0].dyn_cast<ir::Int32_tAttribute>().data() <<
      // std::endl;
    }

    std::cerr << "================================" << std::endl;

    if ((*it)->op_name() == "pd.uniform") {
      InferShapeInterface interface = (*it)->dyn_cast<InferShapeInterface>();
      phi::InferMetaContext ctx;
      ctx.EmplaceBackAttr(phi::IntArray({2, 2}));
      ctx.EmplaceBackAttr(phi::DataType::FLOAT32);

      void* ptr = (*it)->GetResultByIndex(0).impl();
      auto name = name_map[ptr];

      ctx.EmplaceBackOutput(scope.Var(name)->Get<phi::DenseTensor>());
      interface.InferShape(&ctx);

      paddle::dialect::GetOpInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::GetOpInfoInterface>();
      auto op_info_res = op_info_interface.GetOpInfo();

      auto input_info = std::get<0>(op_info_res);

      for (auto& t : input_info) {
        std::cerr << t.name << "\t" << t.type_name << std::endl;
      }

      auto attr_info = std::get<1>(op_info_res);
      for (auto& t : attr_info) {
        std::cerr << t.name << "\t" << t.type_name << std::endl;
      }

      auto runtime_info = std::get<3>(op_info_res);
      std::cerr << "infer meta name " << runtime_info.infer_meta_func
                << std::endl;

      for (auto& t : runtime_info.infer_meta_param) {
        std::cerr << t << std::endl;
      }

      auto phi_kernels =
          phi::KernelFactory::Instance().SelectKernelMap("uniform");
      phi::KernelKey kernel_key(phi::TransToPhiBackend(cpu_place),
                                phi::DataLayout::ALL_LAYOUT,
                                phi::DataType::FLOAT32);
      auto found_it = phi_kernels.find(kernel_key);
      if (found_it == phi_kernels.end()) {
        std::cerr << "not found kernel " << kernel_key << std::endl;

        for (auto it2 = phi_kernels.begin(); it2 != phi_kernels.end(); ++it2) {
          std::cerr << it2->first << "\t" << it2->second << std::endl;
        }
        throw std::runtime_error("not found");
      } else {
        phi::KernelContext kernel_ctx(dev_ctx);

        kernel_ctx.EmplaceBackAttr(phi::IntArray({2, 2}));
        kernel_ctx.EmplaceBackAttr(phi::DataType::FLOAT32);
        kernel_ctx.EmplaceBackAttr(phi::Scalar(0.0));
        kernel_ctx.EmplaceBackAttr(phi::Scalar(1.0));
        kernel_ctx.EmplaceBackAttr(static_cast<int>(2));
        kernel_ctx.EmplaceBackOutput(
            scope.Var(name)->GetMutable<phi::DenseTensor>());

        std::cerr << "begin to run kernel" << std::endl;
        found_it->second(&kernel_ctx);
        std::cerr << "fin kernel" << std::endl;
        std::cerr << scope.Var(name)->Get<phi::DenseTensor>() << std::endl;
      }
    }

    if ((*it)->op_name() == "pd.add") {
      InferShapeInterface interface = (*it)->dyn_cast<InferShapeInterface>();
      phi::InferMetaContext ctx;

      void* ptr = (*it)->GetResultByIndex(0).impl();
      auto name = name_map[ptr];
      auto in_name1 =
          name_map[(*it)->GetOperandByIndex(0).impl()->source().impl()];
      auto in_name2 =
          name_map[(*it)->GetOperandByIndex(1).impl()->source().impl()];

      ctx.EmplaceBackInput(scope.Var(in_name1)->Get<phi::DenseTensor>());
      ctx.EmplaceBackInput(scope.Var(in_name2)->Get<phi::DenseTensor>());
      ctx.EmplaceBackOutput(scope.Var(name)->Get<phi::DenseTensor>());
      interface.InferShape(&ctx);

      paddle::dialect::GetOpInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::GetOpInfoInterface>();

      auto op_info_res = op_info_interface.GetOpInfo();

      auto input_info = std::get<0>(op_info_res);

      for (auto& t : input_info) {
        std::cerr << t.name << "\t" << t.type_name << std::endl;
      }

      auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap("add");
      phi::KernelKey kernel_key(phi::TransToPhiBackend(cpu_place),
                                phi::DataLayout::ALL_LAYOUT,
                                phi::DataType::FLOAT32);
      auto found_it = phi_kernels.find(kernel_key);
      if (found_it == phi_kernels.end()) {
        std::cerr << "not found kernel " << kernel_key << std::endl;

        for (auto it2 = phi_kernels.begin(); it2 != phi_kernels.end(); ++it2) {
          std::cerr << it2->first << "\t" << it2->second << std::endl;
        }
        throw std::runtime_error("not found");
      } else {
        phi::KernelContext kernel_ctx(dev_ctx);

        kernel_ctx.EmplaceBackInput(
            &(scope.Var(in_name1)->Get<phi::DenseTensor>()));
        kernel_ctx.EmplaceBackInput(
            &(scope.Var(in_name2)->Get<phi::DenseTensor>()));

        kernel_ctx.EmplaceBackOutput(
            scope.Var(name)->GetMutable<phi::DenseTensor>());

        std::cerr << "begin to run kernel" << std::endl;
        found_it->second(&kernel_ctx);
        std::cerr << "fin kernel" << std::endl;
        std::cerr << scope.Var(name)->Get<phi::DenseTensor>() << std::endl;
      }
    }
  }
}

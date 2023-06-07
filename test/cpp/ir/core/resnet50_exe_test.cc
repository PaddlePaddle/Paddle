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
#include <chrono>
#include <iostream>
#include <map>
#include <string>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/dialect.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"

#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/ir/core/phi_kernel_adaptor.h"

PD_DECLARE_KERNEL(uniform, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(gaussian, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(feed_dense_tensor, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(conv2d, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(batch_norm, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(pool2d, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cross_entropy_with_softmax, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(topk, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(accuracy, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(scale, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(flatten, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(momentum, CPU, ALL_LAYOUT);

PD_DECLARE_KERNEL(conv2d_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(batch_norm_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(relu_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(pool2d_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(reshape_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cross_entropy_with_softmax_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(mean_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(topk_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(flatten_grad, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(matmul_grad, CPU, ALL_LAYOUT);

using PaddleDialect = paddle::dialect::PaddleDialect;
using ProgramDesc = paddle::framework::ProgramDesc;
using BlockDesc = paddle::framework::BlockDesc;
using OpDesc = paddle::framework::OpDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;

ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);

  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  return ProgramDesc(buffer);
}

TEST(PaddleDialectTest, Translator) {
  auto p = load_from_file("resnet50_startup.prog");
  // auto p = load_from_file( "simple_add_startup_program" );
  EXPECT_EQ(p.Size(), 1u);
  std::cerr << "after load" << std::endl;

  ir::IrContext* ctx = ir::IrContext::Instance();
  ctx->GetOrRegisterDialect<PaddleDialect>();
  ctx->GetOrRegisterDialect<ir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p, true);

  program->Print(std::cout);

  // auto& para_map = program->parameters();

  // for( auto it = para_map.begin(); it != para_map.end(); ++it)
  // {
  //   std::cerr << "param name " <<  it->first <<  std::endl;
  // }

  paddle::framework::Scope scope;

  PhiKernelAdaptor phi_kernel_adaptor(&scope);

  phi_kernel_adaptor.run(program.get());

  std::cerr << "fin run startup program" << std::endl;

  auto& para_map = program->parameters();

  for (auto it = para_map.begin(); it != para_map.end(); ++it) {
    std::cerr << it->first << std::endl;
    if (scope.Var(it->first)->IsInitialized() &&
        scope.Var(it->first)->Get<phi::DenseTensor>().IsInitialized()) {
      std::cerr << "shape "
                << scope.Var(it->first)->Get<phi::DenseTensor>().dims()
                << std::endl;
    }
  }

  auto main_program_desc =
      load_from_file("resnet50_main_no_merged_momentum.prog");

  auto main_program =
      paddle::TranslateLegacyProgramToProgram(main_program_desc);

  // main_program->Print(std::cout);
  // hard code here , only for resnet 50, need update here, init data and label
  // here
  phi::Place cpu_place(phi::AllocationType::CPU);
  auto data_tensor = scope.Var("data")->GetMutable<phi::DenseTensor>();

  data_tensor->set_meta(
      phi::DenseTensorMeta(phi::DataType::FLOAT32, {2, 3, 224, 224}));
  data_tensor->mutable_data(cpu_place, 2 * 3 * 224 * 224 * sizeof(float));

  auto label_tensor = scope.Var("label")->GetMutable<phi::DenseTensor>();

  label_tensor->set_meta(phi::DenseTensorMeta(phi::DataType::INT64, {2}));
  label_tensor->mutable_data(cpu_place, 2 * sizeof(int64_t));

  void* label_ptr = label_tensor->data();
  int64_t* label_int_ptr = dynamic_cast<int64_t*>(label_ptr);
  label_int_ptr[0] = 1;
  label_int_ptr[1] = 1;

  phi_kernel_adaptor.run(main_program.get());
}

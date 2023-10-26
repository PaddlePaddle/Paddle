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
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/ir_adaptor/translator/utils.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/dialect.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/ir_printer.h"
#include "paddle/pir/core/parser/ir_parser.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/dialect/control_flow/ir/cf_ops.h"

using OperatorDialect = paddle::dialect::OperatorDialect;
using ProgramDesc = paddle::framework::ProgramDesc;
using BlockDesc = paddle::framework::BlockDesc;
using OpDesc = paddle::framework::OpDesc;
using VarDesc = paddle::framework::VarDesc;
using VarType = paddle::framework::proto::VarType;

ProgramDesc load_from_file(const std::string &file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);

  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());  // NOLINT
  fin.close();
  return ProgramDesc(buffer);
}

TEST(OperatorDialectTest, MainProgram) {
  auto p = load_from_file("resnet50_main.prog");
  EXPECT_EQ(p.Size(), 1u);

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);

  // ops.size() = op size in BlockDesc + get_parameter_op + combine op + int
  // array op + full op (Note: p already has a full)
  EXPECT_EQ(program->block()->size(),
            p.Block(0).OpSize() + program->parameters_num() + 20 + 5 + 8);
  EXPECT_GT(ss.str().size(), 0u);
}

TEST(OperatorDialectTest, ConditionBlock) {
  auto p = load_from_file("conditional_block_test.prog");
  EXPECT_EQ(p.Size(), 7u);

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  EXPECT_EQ(program->block()->size(), 4u);
  size_t id = 0;
  for (auto &op : *program->block()) {
    if (id == 0 || id == 1) {
      EXPECT_EQ(op->isa<paddle::dialect::FullOp>(), true);
    }
    if (id == 2) {
      EXPECT_EQ(op->isa<paddle::dialect::LessThanOp>(), true);
    }
    if (id == 3) {
      EXPECT_EQ(op->isa<paddle::dialect::IfOp>(), true);
      EXPECT_EQ(op->num_regions(), 2u);
      // true block
      pir::Block *true_block =
          op->dyn_cast<paddle::dialect::IfOp>().true_block();
      size_t true_id = 0;
      for (auto &op1 : *true_block) {
        if (true_id == 0 || true_id == 1) {
          EXPECT_EQ(op1->isa<paddle::dialect::FullOp>(), true);
        }
        if (true_id == 2) {
          EXPECT_EQ(op1->isa<paddle::dialect::LessThanOp>(), true);
        }
        if (true_id == 3) {
          pir::Block *true_true_block =
              op1->dyn_cast<paddle::dialect::IfOp>().true_block();
          size_t true_true_id = 0;
          for (auto &op2 : *true_true_block) {
            if (true_true_id == 0) {
              EXPECT_EQ(op2->isa<paddle::dialect::AddOp>(), true);
            }
            if (true_true_id == 1) {
              EXPECT_EQ(op2->isa<pir::YieldOp>(), true);
            }
            true_true_id++;
          }
          pir::Block *false_false_block =
              op1->dyn_cast<paddle::dialect::IfOp>().false_block();
          size_t false_false_id = 0;
          for (auto &op2 : *false_false_block) {
            if (false_false_id == 0) {
              EXPECT_EQ(op2->isa<paddle::dialect::MultiplyOp>(), true);
            }
            if (false_false_id == 1) {
              EXPECT_EQ(op2->isa<pir::YieldOp>(), true);
            }
            false_false_id++;
          }
        }
        if (true_id == 4) {
          EXPECT_EQ(op1->isa<paddle::dialect::MultiplyOp>(), true);
        }
        if (true_id == 5) {
          EXPECT_EQ(op1->isa<pir::YieldOp>(), true);
        }
        true_id++;
      }
      // false block
      pir::Block *false_block =
          op->dyn_cast<paddle::dialect::IfOp>().false_block();
      size_t false_id = 0;
      for (auto &op1 : *false_block) {
        if (false_id == 0 || false_id == 1) {
          EXPECT_EQ(op1->isa<paddle::dialect::FullOp>(), true);
        }
        if (false_id == 2) {
          EXPECT_EQ(op1->isa<paddle::dialect::LessThanOp>(), true);
        }
        if (false_id == 3) {
          EXPECT_EQ(op1->isa<paddle::dialect::IfOp>(), true);
          // true block
          pir::Block *false_true_block =
              op1->dyn_cast<paddle::dialect::IfOp>().true_block();
          size_t false_true_id = 0;
          for (auto &op2 : *false_true_block) {
            if (false_true_id == 0) {
              EXPECT_EQ(op2->isa<paddle::dialect::AddOp>(), true);
            }
            if (false_true_id == 1) {
              EXPECT_EQ(op2->isa<pir::YieldOp>(), true);
            }
            false_true_id++;
          }
          // false block
          pir::Block *false_false_block =
              op1->dyn_cast<paddle::dialect::IfOp>().true_block();
          size_t false_false_id = 0;
          for (auto &op2 : *false_false_block) {
            if (false_false_id == 0) {
              EXPECT_EQ(op2->isa<paddle::dialect::AddOp>(), true);
            }
            if (false_false_id == 1) {
              EXPECT_EQ(op2->isa<pir::YieldOp>(), true);
            }
            false_false_id++;
          }
        }
        if (false_id == 4) {
          EXPECT_EQ(op1->isa<paddle::dialect::MultiplyOp>(), true);
        }
        if (false_id == 5) {
          EXPECT_EQ(op1->isa<pir::YieldOp>(), true);
        }
        false_id++;
      }
    }
    id++;
  }
}

TEST(OperatorDialectTest, StartupProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  EXPECT_EQ(p.Size(), 1u);

  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  size_t op_size = program->block()->size();
  // ops.size() = op size in BlockDesc + get_parameter_op +
  // consant_op_for_uniform
  // + consant_op for guassian
  EXPECT_EQ(op_size, p.Block(0).OpSize() + program->parameters_num() + 3 + 53);

  std::stringstream ss;
  program->Print(ss);
  EXPECT_GT(ss.str().size(), 0u);
}

TEST(RegisterInfoTest, MainProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  pir::IrContext *ctx = pir::IrContext::Instance();

  auto unregistered_ops =
      paddle::translator::CheckUnregisteredOperation(ctx, p);
  EXPECT_EQ(unregistered_ops.size(), 0u);

  auto new_op = std::unique_ptr<OpDesc>(
      new OpDesc("something must not be registered", {}, {}, {}));
  auto *block = p.MutableBlock(0);
  block->AppendAllocatedOp(std::move(new_op));

  unregistered_ops = paddle::translator::CheckUnregisteredOperation(ctx, p);
  EXPECT_EQ(unregistered_ops.size(), 1u);
  EXPECT_EQ(unregistered_ops[0], "something must not be registered");
}

TEST(IrParserTest, MainProgram) {
  auto p = load_from_file("resnet50_main.prog");
  EXPECT_EQ(p.Size(), 1u);
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);
  std::unique_ptr<pir::Program> parser_program = pir::Program::Parse(ss, ctx);
  std::stringstream ssp;
  parser_program->Print(ssp);

  EXPECT_TRUE(ssp.str() == ss.str());
}

TEST(IrParserTest, StartupProgram) {
  auto p = load_from_file("resnet50_startup.prog");
  EXPECT_EQ(p.Size(), 1u);
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);
  std::unique_ptr<pir::Program> parser_program = pir::Program::Parse(ss, ctx);
  std::stringstream ssp;
  parser_program->Print(ssp);

  EXPECT_TRUE(ssp.str() == ss.str());
}

TEST(OperatorDialectTest, WhileOpProgram) {
  auto p = load_from_file("while_op_test.prog");
  EXPECT_EQ(p.Size(), 3u);
  pir::IrContext *ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  std::stringstream ss;
  program->Print(ss);

  LOG(INFO) << ss.str();

  EXPECT_EQ(program->block()->size(), 4u);
  size_t id = 0;
  for (auto &op : *program->block()) {
    if (id == 0 || id == 1) {
      EXPECT_TRUE(op->isa<paddle::dialect::FullOp>());
    }
    if (id == 2) {
      EXPECT_TRUE(op->isa<paddle::dialect::LessThanOp>());
    }
    if (id == 3) {
      EXPECT_TRUE(op->isa<paddle::dialect::WhileOp>());
      EXPECT_EQ(op->num_regions(), 1u);
      // body block
      pir::Block *body_block =
          op->dyn_cast<paddle::dialect::WhileOp>().body_block();
      size_t body_id = 0;
      for (auto &op1 : *body_block) {
        if (body_id == 0) {
          EXPECT_TRUE(op1->isa<paddle::dialect::FullOp>());
        }
        if (body_id == 1) {
          EXPECT_TRUE(op1->isa<paddle::dialect::ScaleOp>());
        }
        if (body_id == 2) {
          EXPECT_TRUE(op1->isa<paddle::dialect::LessThanOp>());
        }
        if (body_id == 3) {
          pir::Block *body_body_block =
              op1->dyn_cast<paddle::dialect::WhileOp>().body_block();
          size_t body_body_id = 0;
          for (auto &op2 : *body_body_block) {
            if (body_body_id == 0) {
              EXPECT_TRUE(op2->isa<paddle::dialect::FullOp>());
            }
            if (body_body_id == 1) {
              EXPECT_TRUE(op2->isa<paddle::dialect::ScaleOp>());
            }
            if (body_body_id == 2) {
              EXPECT_TRUE(op2->isa<paddle::dialect::LessThanOp>());
            }
            if (body_body_id == 3) {
              EXPECT_TRUE(op2->isa<pir::YieldOp>());
            }
            body_body_id++;
          }
        }
        if (body_id == 4) {
          EXPECT_TRUE(op1->isa<paddle::dialect::LessThanOp>());
        }
        if (body_id == 5) {
          EXPECT_TRUE(op1->isa<pir::YieldOp>());
        }
        body_id++;
      }
    }
    id++;
  }
}

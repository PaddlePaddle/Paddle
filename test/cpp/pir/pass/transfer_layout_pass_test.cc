// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "paddle/common/layout.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/inference/api/paddle_pass_builder.h"
#include "paddle/fluid/ir_adaptor/translator/translate.h"
#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/transforms/general/transfer_layout_pass.h"
#include "paddle/fluid/pir/transforms/passes.h"
#include "paddle/phi/core/framework/framework.pb.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/ir_context.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/pass/pass_manager.h"

using ProgramDesc = paddle::framework::ProgramDesc;
ProgramDesc load_from_file(const std::string& file_name) {
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);

  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());  // NOLINT
  fin.close();
  return ProgramDesc(buffer);
}

TEST(transfer_layout_pass, pass_test) {
  // Load Unet Program
  const std::string model_name = "sd15_unet.pdmodel";
  auto p = load_from_file(model_name);
  EXPECT_EQ(p.Size(), 1u);
  EXPECT_GT(p.Block(0).OpSize(), 0u);

  // Translate to PIR Program
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::BuiltinDialect>();
  auto program = paddle::TranslateLegacyProgramToProgram(p);

  pir::PassManager pass_pm(::pir::IrContext::Instance(), 3);

  // Note(lyk) To avoid windows compiler error:
  // paddle::kPirGpuPasses has already been declared as
  // PD_INFER_DECL, but we still got LINK ERROR. The
  // reason is still unclear, skip it now.
  const std::vector<std::string> kCopiedPirGpuPasses{
      // Functional pass
      "map_op_to_another_pass",
      "identity_op_clean_pass",
      // Operator fusion pass
      "silu_fuse_pass",
      "conv2d_bn_fuse_pass",
      "conv2d_add_act_fuse_pass",
      "conv2d_add_fuse_pass",
      "embedding_eltwise_layernorm_fuse_pass",
      "fused_flash_attn_pass",
      "multihead_matmul_fuse_pass",
      "matmul_add_act_fuse_pass",
      "fc_elementwise_layernorm_fuse_pass",
      "matmul_scale_fuse_pass",
      "matmul_transpose_fuse_pass",
      "transpose_flatten_concat_fuse_pass",
      "remove_redundant_transpose_pass"};

  for (const auto& gpu_pass : kCopiedPirGpuPasses) {
    pass_pm.AddPass(pir::PassRegistry::Instance().Get(gpu_pass));
  }
  pass_pm.Run(program.get());

  pir::PassManager transfer_layout_manager(::pir::IrContext::Instance(), 4);
  transfer_layout_manager.AddPass(pir::CreateTransferLayoutPass());
  transfer_layout_manager.Run(program.get());
}

// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/inference/lite/op_teller.h"

namespace paddle {
namespace inference {
namespace analysis {
namespace lite {
void StrToBinaryFile(const std::string& path, const std::string& str);
void ModifyHostSubgraphOps(framework::ProgramDesc* host_program,
                           framework::BlockDesc* host_sub_block,
                           const std::vector<framework::OpDesc*>& subgraph_ops);
void AppendLiteSubBlocks(const std::vector<framework::OpDesc*>& subgraph_ops,
                         framework::ProgramDesc* engine_program,
                         framework::ProgramDesc* host_program,
                         const int32_t host_sub_id);
}

TEST(LiteSubgraphPass, basic) {
  framework::ProgramDesc host_program;
  framework::ProgramDesc engine_program;
  framework::BlockDesc* host_main_block = host_program.MutableBlock(0);
  framework::BlockDesc* host_sub_block =
      host_program.AppendBlock(*host_main_block);
  framework::OpDesc* host_while_op = host_main_block->AppendOp();
  host_main_block->Var("var_main");
  host_sub_block->Var("var_sub");
  host_while_op->SetType("while");
  host_while_op->SetAttr("sub_block", host_sub_block);
  framework::OpDesc* host_sub_block_op = host_sub_block->AppendOp();
  host_sub_block_op->SetType("leaky_relu");

  CHECK(inference::lite::OpTeller::Global().Tell("while", *host_while_op))
      << "Lite operator teller test failed.";

  lite::AppendLiteSubBlocks({host_while_op}, &engine_program, &host_program,
                            host_sub_block->ID());
  lite::ModifyHostSubgraphOps(&host_program, host_sub_block, {host_while_op});
  lite::StrToBinaryFile("./", "test");
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

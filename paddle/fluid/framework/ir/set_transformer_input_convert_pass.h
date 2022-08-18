// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <utility>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

//     in_var  emb
//       |      |
//     lookup_table
//           |
//        lkt_var

//
struct SetTransformerInputConvert : public PatternBase {
  SetTransformerInputConvert(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "transformer_input_convert_pass") {}
  void operator()(const std::string &pos_id);
  // declare operator node's name
  PATTERN_DECL_NODE(lookup_table);
  // declare variable node's name
  PATTERN_DECL_NODE(lookup_table_id);
};

struct MultiheadMatmulOP : public PatternBase {
  MultiheadMatmulOP(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "transformer_input_convert_pass") {}
  void operator()();
  // declare operator node's name
  PATTERN_DECL_NODE(multihead_matmul);
  PATTERN_DECL_NODE(multihead_matmul_out);
};
}  // namespace patterns

class SetTransformerInputConvertPass : public FusePassBase {
 public:
  SetTransformerInputConvertPass() {}
  virtual ~SetTransformerInputConvertPass() {}

 protected:
  void ApplyImpl(Graph *graph) const;
  const std::string name_scope_{"transformer_input_convert_pass"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

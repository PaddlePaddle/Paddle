/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/trt_llm_rotary_attention_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/inference/tensorrt/helper.h"
#endif

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct RotaryAttention : public PatternBase {
  RotaryAttention(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "rotary_attention") {}

  void operator()(PDNode *x, PDNode *y);

  // declare operator node's name
  PATTERN_DECL_NODE(elementwise);
};

void RotaryAttention::operator()(PDNode *x, PDNode *y) {
  // Create nodes for elementwise add op.
}

}  // namespace patterns

void TrtLLMRotaryAttentionPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("rotary_attention_fuse", graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(trt_llm_rotary_attention_pass,
              paddle::framework::ir::TrtLLMRotaryAttentionPass);

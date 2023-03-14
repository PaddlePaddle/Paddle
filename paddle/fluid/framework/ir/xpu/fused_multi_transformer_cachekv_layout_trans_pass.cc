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

#include <string>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/ir/xpu/quant_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/kernels/concat_kernel.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace {
const int fill_constant_shape_tensor_list_names_size = 5;
}

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {
struct FusedMultiTransformerCacheKVLayoutTransPattern : public PatternBase {
  FusedMultiTransformerCacheKVLayoutTransPattern(PDPattern* pattern,
                                                 const std::string& name_scope);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_multi_transformer);
  PATTERN_DECL_NODE(fill_constant_0);
  PATTERN_DECL_NODE(fill_constant_1);
  PATTERN_DECL_NODE(fill_constant_2);
  PATTERN_DECL_NODE(fill_constant_3);
  PATTERN_DECL_NODE(fill_constant_reduce);
  PATTERN_DECL_NODE(shape);
  PATTERN_DECL_NODE(slice);
  PATTERN_DECL_NODE(fill_constant_0_out);
  PATTERN_DECL_NODE(fill_constant_1_out);
  PATTERN_DECL_NODE(fill_constant_2_out);
  PATTERN_DECL_NODE(fill_constant_3_out);
  PATTERN_DECL_NODE(fill_constant_reduce_out);
  PATTERN_DECL_NODE(shape_in);
  PATTERN_DECL_NODE(shape_out);
  PATTERN_DECL_NODE(slice_out);
};  // struct FusedMultiTransformerCacheKVLayoutTransPattern

FusedMultiTransformerCacheKVLayoutTransPattern::
    FusedMultiTransformerCacheKVLayoutTransPattern(
        PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  // fill_constant_0 -> 2
  auto* fill_constant_0 =
      pattern->NewNode(fill_constant_0_repr())->assert_is_op("fill_constant");
  auto* fill_constant_0_out =
      pattern->NewNode(fill_constant_0_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_nth_input("fill_constant", "ShapeTensorList", 0);
  fill_constant_0->LinksTo({fill_constant_0_out});
  // shape + slice -> batch
  auto* shape_in = pattern->NewNode(shape_in_repr())
                       ->assert_is_op_input("shape", "Input")
                       ->AsInput();
  auto* shape = pattern->NewNode(shape_repr())->assert_is_op("shape");
  auto* shape_out =
      pattern->NewNode(shape_out_repr())->assert_is_op_output("shape", "Out");
  auto* slice = pattern->NewNode(slice_repr())->assert_is_op("slice");
  auto* slice_out =
      pattern->NewNode(slice_out_repr())
          ->assert_is_op_output("slice", "Out")
          ->assert_is_op_nth_input("fill_constant", "ShapeTensorList", 1);
  shape->LinksFrom({shape_in}).LinksTo({shape_out});
  slice->LinksFrom({shape_out}).LinksTo({slice_out});
  // fill_constant_1 -> head_num
  auto* fill_constant_1 =
      pattern->NewNode(fill_constant_1_repr())->assert_is_op("fill_constant");
  auto* fill_constant_1_out =
      pattern->NewNode(fill_constant_1_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_nth_input("fill_constant", "ShapeTensorList", 2);
  fill_constant_1->LinksTo({fill_constant_1_out});
  // fill_constant_2 -> max_seqlen
  auto* fill_constant_2 =
      pattern->NewNode(fill_constant_2_repr())->assert_is_op("fill_constant");
  auto* fill_constant_2_out =
      pattern->NewNode(fill_constant_2_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_nth_input("fill_constant", "ShapeTensorList", 3);
  fill_constant_2->LinksTo({fill_constant_2_out});
  // fill_constant_3 -> head_dim
  auto* fill_constant_3 =
      pattern->NewNode(fill_constant_3_repr())->assert_is_op("fill_constant");
  auto* fill_constant_3_out =
      pattern->NewNode(fill_constant_3_out_repr())
          ->assert_is_op_output("fill_constant", "Out")
          ->assert_is_op_nth_input("fill_constant", "ShapeTensorList", 4);
  fill_constant_3->LinksTo({fill_constant_3_out});
  // fill_constant_reduce
  auto* fill_constant_reduce = pattern->NewNode(fill_constant_reduce_repr())
                                   ->assert_is_op("fill_constant");
  auto* fill_constant_reduce_out =
      pattern->NewNode(fill_constant_reduce_out_repr())
          ->assert_is_op_output("fill_constant", "Out");
  fill_constant_reduce
      ->LinksFrom({fill_constant_0_out,
                   fill_constant_1_out,
                   fill_constant_2_out,
                   fill_constant_3_out,
                   slice_out})
      .LinksTo({fill_constant_reduce_out});
  // fused_multi_trans_former
  auto* fused_multi_trans_former =
      pattern->NewNode(fused_multi_transformer_repr())
          ->assert_is_op("fused_multi_transformer");
  fused_multi_trans_former->LinksFrom({fill_constant_reduce_out});
}
}  // namespace patterns

class FusedMultiTransformerCacheKVLayoutTransPass : public FusePassBase {
 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  const std::string name_scope_{
      "fused_multi_transformer_cachekv_layout_trans_pass"};
};

void FusedMultiTransformerCacheKVLayoutTransPass::ApplyImpl(
    ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  GraphPatternDetector gpd;
  patterns::FusedMultiTransformerCacheKVLayoutTransPattern pattern(
      gpd.mutable_pattern(), name_scope_);
  int found_subgraph_count = 0;
  auto* scope = param_scope();
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle FusedMultiTransformerCacheKVLayoutTransPass";
    GET_IR_NODE(fused_multi_transformer);
    GET_IR_NODE(fill_constant_reduce);
    auto fill_constant_reduce_input_names =
        fill_constant_reduce->Op()->Input("ShapeTensorList");
    if (fill_constant_reduce_input_names.size() !=
        fill_constant_shape_tensor_list_names_size) {
      VLOG(3) << "fill constant Input['ShapeTensorList'] names size should be "
              << fill_constant_shape_tensor_list_names_size << ", but received "
              << fill_constant_reduce_input_names.size();
      return;
    }
    auto fill_constant_reduce_trans_input_names =
        std::vector<std::string>{fill_constant_reduce_input_names[0],
                                 fill_constant_reduce_input_names[3],
                                 fill_constant_reduce_input_names[1],
                                 fill_constant_reduce_input_names[2],
                                 fill_constant_reduce_input_names[4]};
    fill_constant_reduce->Op()->SetInput(
        "ShapeTensorList", fill_constant_reduce_trans_input_names);
    // auto* fused_multi_transformer_op = fused_multi_transformer->Op();
    // const std::string cache_kv_input_name = "CacheKV";
    // auto& inputs_name = fused_multi_transformer_op->InputNames();
    // if (std::find(inputs_name.begin(), inputs_name.end(),
    // cache_kv_input_name) == inputs_name.end()) {
    //     return;
    // }
    // auto cache_kv_input_variable_names =
    // fused_multi_transformer_op->Input(cache_kv_input_name);
    ++found_subgraph_count;
  };
  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(
    fused_multi_transformer_cachekv_layout_trans_pass,
    paddle::framework::ir::FusedMultiTransformerCacheKVLayoutTransPass);

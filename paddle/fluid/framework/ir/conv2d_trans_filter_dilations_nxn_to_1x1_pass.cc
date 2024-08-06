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

#include <string>

#include "glog/logging.h"

#include "paddle/fluid/framework/ir/conv2d_trans_filter_dilations_nxn_to_1x1_pass.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/quantize_helper.h"
#include "paddle/fluid/framework/ir/xpu/pass_utils.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle::framework {
class Scope;
}  // namespace paddle::framework

namespace paddle::framework::ir::patterns {

struct Conv2dLargeDilationsPattern : public PatternBase {
  Conv2dLargeDilationsPattern(PDPattern* pattern,
                              const std::string& name_scope);

  PATTERN_DECL_NODE(conv2d);
};

Conv2dLargeDilationsPattern::Conv2dLargeDilationsPattern(
    PDPattern* pattern, const std::string& name_scope)
    : PatternBase(pattern, name_scope, name_scope) {
  pattern->NewNode(conv2d_repr())
      ->assert_is_op("conv2d")
      ->assert_more([](Node* node) {
        auto data_format =
            node->Op()->GetAttrIfExists<std::string>("data_format");
        if (data_format != "NCHW") return false;
        auto dilations =
            node->Op()->GetAttrIfExists<std::vector<int>>("dilations");
        if (dilations.size() != 2) return false;
        return dilations[0] * dilations[1] > 1;
      });
}

}  // namespace paddle::framework::ir::patterns
namespace paddle::framework::ir {

void Conv2dTransFilterDilationsNxNTo1x1Pass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::PreconditionNotMet("graph should not be null."));
  Init(name_scope_, graph);
  conv2d_dilation_trans(graph);
}
template <class T>
static void conv2d_dilation_trans_fn(const T* weights_data,
                                     T* new_weights_data,
                                     uint64_t kn,
                                     uint64_t kc,
                                     uint64_t kh,
                                     uint64_t kw,
                                     uint64_t new_kh,
                                     uint64_t new_kw,
                                     uint64_t dilation_h,
                                     uint64_t dilation_w) {
  for (uint64_t n = 0; n < kn; n++) {
    for (uint64_t c = 0; c < kc; c++) {
      for (uint64_t h = 0; h < kh; h++) {
        uint64_t h_offset = dilation_h * h;
        for (uint64_t w = 0; w < kw; w++) {
          uint64_t w_offset = dilation_w * w;
          uint64_t new_offset = n * kc * new_kh * new_kw + c * new_kh * new_kw +
                                h_offset * new_kw + w_offset;
          uint64_t old_offset = n * kc * kh * kw + c * kh * kw + h * kw + w;
          new_weights_data[new_offset] = weights_data[old_offset];
        }
      }
    }
  }
}

void Conv2dTransFilterDilationsNxNTo1x1Pass::conv2d_dilation_trans(
    ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::Conv2dLargeDilationsPattern pattern(gpd.mutable_pattern(),
                                                name_scope_);
  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* graph) {
    VLOG(4) << "handle conv2d large dilation trans";
    GET_IR_NODE_FROM_SUBGRAPH(conv2d, conv2d, pattern);
    auto* block = conv2d->Op()->Block();
    auto* scope = param_scope();
    auto weights_name = conv2d->Op()->Input("Filter")[0];
    auto dilations =
        conv2d->Op()->GetAttrIfExists<std::vector<int>>("dilations");
    // For kunlunxin, if the dilation is bigger than 8, xdnn will occur
    // XDNN_NO_ENOUGH_WORKSPACE error.
    if (dilations[0] >= 8 || dilations[1] >= 8) return;
    auto* weights =
        scope->FindVar(weights_name)->GetMutable<phi::DenseTensor>();
    auto weights_shape = weights->dims();
    int kh = static_cast<int>(weights_shape[2]);
    int kw = static_cast<int>(weights_shape[3]);
    int new_kh = static_cast<int>(dilations[0] * (kh - 1) + 1);
    int new_kw = static_cast<int>(dilations[1] * (kw - 1) + 1);
    // New weights
    auto new_weights_name = weights_name + "_dilation_trans";
    auto* new_weights =
        scope->Var(new_weights_name)->GetMutable<phi::DenseTensor>();
    new_weights->Resize({weights_shape[0], weights_shape[1], new_kh, new_kw});
    auto* cpu_ctx = static_cast<phi::CPUContext*>(
        phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
    if (weights->dtype() == phi::DataType::FLOAT32) {
      auto weights_data = weights->data<float>();
      auto* new_weights_data = cpu_ctx->Alloc<float>(new_weights);
      memset(new_weights_data, 0, new_weights->numel() * sizeof(float));
      conv2d_dilation_trans_fn<float>(weights_data,
                                      new_weights_data,
                                      static_cast<int>(weights_shape[0]),
                                      static_cast<int>(weights_shape[1]),
                                      kh,
                                      kw,
                                      new_kh,
                                      new_kw,
                                      dilations[0],
                                      dilations[1]);
    } else if (weights->dtype() == phi::DataType::FLOAT16) {
      auto weights_data = weights->data<phi::dtype::float16>();
      auto* new_weights_data = cpu_ctx->Alloc<phi::dtype::float16>(new_weights);
      memset(new_weights_data,
             0,
             new_weights->numel() * sizeof(phi::dtype::float16));
      conv2d_dilation_trans_fn<phi::dtype::float16>(
          weights_data,
          new_weights_data,
          static_cast<int>(weights_shape[0]),
          static_cast<int>(weights_shape[1]),
          kh,
          kw,
          new_kh,
          new_kw,
          dilations[0],
          dilations[1]);
    } else if (weights->dtype() == phi::DataType::INT8) {
      auto weights_data = weights->data<int8_t>();
      auto* new_weights_data = cpu_ctx->Alloc<int8_t>(new_weights);
      memset(new_weights_data, 0, new_weights->numel() * sizeof(int8_t));
      conv2d_dilation_trans_fn<int8_t>(weights_data,
                                       new_weights_data,
                                       static_cast<int>(weights_shape[0]),
                                       static_cast<int>(weights_shape[1]),
                                       kh,
                                       kw,
                                       new_kh,
                                       new_kw,
                                       dilations[0],
                                       dilations[1]);
      // Update new weights quant scales
      std::unordered_map<std::string, std::vector<float>> var_quant_scales =
          GetQuantInfoFromTheGraph(graph, "has_quant_info", "var_quant_scales");
      std::unordered_map<std::string, std::vector<float>> new_weights_scale{
          {new_weights_name, var_quant_scales[weights_name]}};
      SaveQuantInfoInTheGraph(
          graph, "has_quant_info", "var_quant_scales", new_weights_scale);
    } else {
      VLOG(3) << "Transfilter only support float32/float16/int8 dtype of "
                 "weights -- do "
                 "nothing and break.";
      return;
    }

    VarDesc new_weights_desc(new_weights_name);
    new_weights_desc.SetPersistable(true);
    new_weights_desc.SetShape(common::vectorize(new_weights->dims()));
    new_weights_desc.SetDataType(
        framework::TransToProtoVarType(new_weights->dtype()));
    auto* new_weights_node = graph->CreateVarNode(&new_weights_desc);
    auto* block_new_weights_desc = block->Var(new_weights_name);
    block_new_weights_desc->SetPersistable(new_weights_desc.Persistable());
    block_new_weights_desc->SetShape(new_weights_desc.GetShape());
    block_new_weights_desc->SetDataType(new_weights_desc.GetDataType());
    // Update conv2d node
    conv2d->Op()->SetAttr("dilations", std::vector<int>({1, 1}));
    conv2d->Op()->RenameInput(weights_name, new_weights_name);
    IR_NODE_LINK_TO(new_weights_node, conv2d);
    LOG(INFO) << "Transfilter dilations from " << Vec2Str<int>(dilations)
              << " to (1, 1) successfully, ori_weights: " << weights_name << "["
              << weights->dims() << "], new_weights: " << new_weights_name
              << "[" << new_weights->dims() << "]";
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(conv2d_trans_filter_dilations_nxn_to_1x1_pass,
              paddle::framework::ir::Conv2dTransFilterDilationsNxNTo1x1Pass);
REGISTER_PASS_CAPABILITY(conv2d_trans_filter_dilations_nxn_to_1x1_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "conv2d", 1));

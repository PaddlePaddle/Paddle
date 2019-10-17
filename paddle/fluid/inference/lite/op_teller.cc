// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/lite/op_teller.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace inference {
namespace lite {

// Just tell by the op_types.
struct SimpleOpTeller : public Teller {
  SimpleOpTeller() {
    ops_.insert("fake_dequantize_max_abs");
    ops_.insert("squeeze");
    ops_.insert("squeeze2");
    ops_.insert("softmax");
    ops_.insert("shuffle_channel");
    ops_.insert("graph_op");
    ops_.insert("generate_proposals");
    ops_.insert("transpose");
    ops_.insert("transpose2");
    ops_.insert("axpy");
    ops_.insert("affine_channel");
    ops_.insert("feed");
    ops_.insert("fill_constant");
    ops_.insert("io_copy");
    ops_.insert("batch_norm");
    ops_.insert("unsqueeze");
    ops_.insert("unsqueeze2");
    ops_.insert("mean");
    ops_.insert("calib_once");
    ops_.insert("fake_quantize_range_abs_max");
    ops_.insert("arg_max");
    ops_.insert("conv2d");
    ops_.insert("depthwise_conv2d");
    ops_.insert("crop");
    ops_.insert("multiclass_nms");
    ops_.insert("roi_align");
    ops_.insert("gru");
    ops_.insert("reshape");
    ops_.insert("reshape2");
    ops_.insert("assign_value");
    ops_.insert("conv2d_transpose");
    ops_.insert("reduce_mean");
    ops_.insert("density_prior_box");
    ops_.insert("fill_constant_batch_size_like");
    ops_.insert("decode_bboxes");
    ops_.insert("prior_box");
    ops_.insert("reduce_max");
    ops_.insert("square");
    ops_.insert("relu");
    ops_.insert("leaky_relu");
    ops_.insert("relu_clipped");
    ops_.insert("prelu");
    ops_.insert("sigmoid");
    ops_.insert("tanh");
    ops_.insert("swish");
    ops_.insert("relu6");
    ops_.insert("log");
    ops_.insert("exp");
    ops_.insert("floor");
    ops_.insert("hard_sigmoid");
    ops_.insert("rsqrt");
    ops_.insert("box_coder");
    ops_.insert("cast");
    ops_.insert("stack");
    ops_.insert("dropout");
    ops_.insert("matmul");
    ops_.insert("calib");
    ops_.insert("flatten");
    ops_.insert("flatten2");
    ops_.insert("shape");
    ops_.insert("lrn");
    ops_.insert("concat");
    ops_.insert("layout");
    ops_.insert("negative");
    ops_.insert("power");
    ops_.insert("range");
    ops_.insert("sequence_expand_as");
    ops_.insert("assign");
    ops_.insert("norm");
    ops_.insert("slice");
    ops_.insert("fetch");
    ops_.insert("box_clip");
    ops_.insert("fc");
    ops_.insert("io_copy_once");
    ops_.insert("elementwise_sub");
    ops_.insert("elementwise_add");
    ops_.insert("elementwise_mul");
    ops_.insert("elementwise_max");
    ops_.insert("elementwise_div");
    ops_.insert("fake_quantize_moving_average_abs_max");
    ops_.insert("im2sequence");
    ops_.insert("anchor_generator");
    ops_.insert("layout_once");
    ops_.insert("expand");
    ops_.insert("scale");
    ops_.insert("uniform_random");
    ops_.insert("gru_unit");
    ops_.insert("pool2d");
    ops_.insert("mul");
    ops_.insert("sequence_expand");
    ops_.insert("yolo_box");
    ops_.insert("split");
    ops_.insert("nearest_interp");
    ops_.insert("bilinear_interp");
    ops_.insert("fusion_elementwise_sub_activation");
    ops_.insert("fusion_elementwise_add_activation");
    ops_.insert("fusion_elementwise_mul_activation");
    ops_.insert("fusion_elementwise_max_activation");
    ops_.insert("fusion_elementwise_div_activation");
    ops_.insert("pad2d");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    return ops_.count(op_type);
  }

 private:
  std::unordered_set<std::string> ops_ {};
};

struct SingleBlockOpTeller : public Teller {
  SingleBlockOpTeller() {
    ops_.insert("while");
  }

  bool operator()(const std::string& op_type,
                  const framework::OpDesc& op_desc) override {
    if (ops_.count(op_type)) {
      SimpleOpTeller supported;
      const int id = op_desc.GetBlockAttrId("sub_block");
      const framework::BlockDesc& block_desc = op_desc.Block()->Program()->Block(id);
      const std::vector<framework::OpDesc *>& ops_sub_block = block_desc.AllOps();
      for (auto* op: ops_sub_block) {
        if (!supported(op->Type(), *op) && !this->operator()(op->Type(), *op)) {
          return false;
        };
      }
      return true;
    }
    return false;
  }

 private:
  std::unordered_set<std::string> ops_;
};


bool OpTeller::Tell(const std::string& op_type, const framework::OpDesc& desc) {
  for (auto& teller : tellers_) {
    if ((*teller)(op_type, desc)) return true;
  }
  return false;
}

OpTeller::OpTeller() {
  tellers_.emplace_back(new SimpleOpTeller);
  tellers_.emplace_back(new SingleBlockOpTeller);
}

}  // namespace lite
}  // namespace inference
}  // namespace paddle



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

#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/data_layout.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

// Just tell by the op_types.
struct SimpleOpTypeSetTeller : public Teller {
  SimpleOpTypeSetTeller() {
#if IS_TRT_VERSION_GE(5130)
    teller_set.insert("relu6");
    teller_set.insert("hard_sigmoid");
    teller_set.insert("clip");
    int8_teller_set.insert("relu6");
    int8_teller_set.insert("hard_sigmoid");
    int8_teller_set.insert("clip");
#endif
#if IS_TRT_VERSION_GE(6000)
    teller_set.insert("fused_embedding_eltwise_layernorm");
    teller_set.insert("multihead_matmul");
    teller_set.insert("skip_layernorm");
    teller_set.insert("slice");
#endif
#if IS_TRT_VERSION_GE(7130)
    teller_set.insert("group_norm");
#endif
  }

  bool operator()(const std::string& op_type, const framework::OpDesc& desc,
                  bool use_no_calib_int8) override {
    if (use_no_calib_int8) {
      return int8_teller_set.count(op_type);
    } else {
      return teller_set.count(op_type);
    }
  }

 private:
  // use this set for no calib int8.
  std::unordered_set<std::string> int8_teller_set{"mul",
                                                  "conv2d",
                                                  "conv2d_fusion",
                                                  "pool2d",
                                                  "relu",
                                                  "depthwise_conv2d",
                                                  "softmax",
                                                  "sigmoid",
                                                  "batch_norm",
                                                  "elementwise_add",
                                                  "leaky_relu",
                                                  "fc",
                                                  "concat",
                                                  "scale",
                                                  "elementwise_mul",
                                                  "conv2d_transpose",
                                                  "hard_swish"};
  std::unordered_set<std::string> teller_set{
      "mul",
      "matmul",
      "conv2d",
      "conv2d_fusion",
      "pool2d",
      "relu",
      "softmax",
      "sigmoid",
      "hard_swish",
      "depthwise_conv2d",
      "batch_norm",
      "concat",
      "tanh",
      "pad",
      "elementwise_add",
      "elementwise_mul",
      "dropout",
      "prelu",
      "conv2d_transpose",
      "leaky_relu",
      "fc",
      "shuffle_channel",
      "swish",
      "split",
      "instance_norm",
      "gelu",
      "layer_norm",
      "scale",
      "stack",
      "transpose2",
      "transpose",
      "flatten2",
      "flatten",
      "gather",
      "yolo_box",
      "roi_align",
      "affine_channel",
      "multiclass_nms",
      "nearest_interp",
      "anchor_generator",
  };
};

bool OpTeller::Tell(const framework::ir::Node* node, bool use_no_calib_int8,
                    bool with_dynamic_shape) {
  const std::string op_type = node->Op()->Type();
  const framework::OpDesc desc = *node->Op();
  // do not support the op which is labeled the `skip_quant`
  if ((desc.HasAttr("namescope") &&
       BOOST_GET_CONST(std::string, desc.GetAttr("op_namescope")) ==
           "/skip_quant_2/") ||
      desc.HasAttr("skip_quant"))
    return false;

  for (auto& teller : tellers_) {
    if (op_type == "pool2d" || op_type == "depthwise_conv2d") {
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));

      if (paddings.size() > 2) return false;
    }

    if (op_type == "conv2d" || op_type == "conv2d_transpose" ||
        op_type == "conv2d_fusion") {
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));

      // conv2d and conv2d_transpose need padding check
      if (paddings.size() > 2 && op_type != "conv2d_fusion") return false;

      if (desc.Input("Input").size() != 1) {
        VLOG(1) << "TRT Conv2d expect 1 input, but got "
                << desc.Input("Input").size() << " input.";
        return false;
      }

      if (desc.Input("Filter").size() != 1) {
        VLOG(1) << "TRT Conv2d expect 1 filter, but got "
                << desc.Input("Filter").size() << " filter.";
        return false;
      }

      if (desc.Output("Output").size() != 1) {
        VLOG(1) << "TRT Conv2d expect 1 output, but got "
                << desc.Output("Output").size() << " output.";
        return false;
      }
    }

    if (op_type == "matmul") {
      auto* block = desc.Block();
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() < 3) {
            VLOG(1)
                << "matmul op dims < 3 not supported in tensorrt, but got dims "
                << shape.size() << ", so jump it.";
            return false;
          }
        }
      }
    }
    if (op_type == "group_norm") {
      if (!with_dynamic_shape) return false;
      bool has_attrs = (desc.HasAttr("epsilon") && desc.HasAttr("groups"));
      if (has_attrs == false) return false;

      auto registry = GetPluginRegistry();
      if (registry == nullptr) return false;
    }
    if (op_type == "concat") {
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
        int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
        if (with_dynamic_shape) {
          if (axis < 0) return false;
        } else {
          if (axis <= 0) return false;
        }
      }
    }
    if (op_type == "transpose2" || op_type == "transpose") {
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
        std::vector<int> axis =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("axis"));
        if (!with_dynamic_shape && axis[0] != 0) return false;
        if (axis.size() >= nvinfer1::Dims::MAX_DIMS) return false;
      }
    }
    if (op_type == "flatten2" || op_type == "flatten") {
      // flatten doesn't support dynamic shape currently
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
        if (with_dynamic_shape) return false;
        int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
        if (axis != 1) return false;
      }
    }

    if (op_type == "gather") {
      // current not support axis from input, use default 0
      if (!with_dynamic_shape || desc.Input("Axis").size() > 0) return false;
    }

    if (op_type == "yolo_box") {
      if (with_dynamic_shape) return false;
      bool has_attrs =
          (desc.HasAttr("class_num") && desc.HasAttr("anchors") &&
           desc.HasAttr("downsample_ratio") && desc.HasAttr("conf_thresh") &&
           desc.HasAttr("clip_bbox") && desc.HasAttr("scale_x_y"));
      if (!has_attrs) return false;
    }

    if (op_type == "affine_channel") {
      if (!desc.HasAttr("data_layout")) return false;
      auto data_layout = framework::StringToDataLayout(
          BOOST_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != framework::DataLayout::kNCHW) return false;
    }

    if (op_type == "multiclass_nms") {
      if (with_dynamic_shape) return false;
      auto* block = desc.Block();
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() != 3) {
            VLOG(1) << "multiclass_nms op dims != 3 not supported in tensorrt, "
                       "but got dims "
                    << shape.size() << ", so jump it.";
            return false;
          }
        }
      }
      bool has_attrs =
          (desc.HasAttr("background_label") &&
           desc.HasAttr("score_threshold") && desc.HasAttr("nms_top_k") &&
           desc.HasAttr("keep_top_k") && desc.HasAttr("normalized"));
      if (has_attrs == false) return false;

      auto nms_top_k = BOOST_GET_CONST(int, desc.GetAttr("nms_top_k"));
      if (nms_top_k < 0) return false;

      auto keep_top_k = BOOST_GET_CONST(int, desc.GetAttr("keep_top_k"));
      if (keep_top_k < 0) return false;

      auto registry = GetPluginRegistry();
      if (registry == nullptr) return false;
    }

    if (op_type == "fc" || op_type == "mul") {
      const int x_num_col_dims =
          desc.HasAttr("x_num_col_dims")
              ? BOOST_GET_CONST(int, desc.GetAttr("x_num_col_dims"))
              : (desc.HasAttr("in_num_col_dims")
                     ? BOOST_GET_CONST(int, desc.GetAttr("in_num_col_dims"))
                     : 1);
      if (x_num_col_dims != 1 && x_num_col_dims != 2) {
        return false;
      }
    }

    if (op_type == "nearest_interp") {
      std::vector<std::string> attrs{"data_layout",   "interp_method",
                                     "align_corners", "scale",
                                     "out_h",         "out_w"};
      for (auto const attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }
      auto data_layout = framework::StringToDataLayout(
          BOOST_GET_CONST(std::string, desc.GetAttr("data_layout")));
      if (data_layout != framework::DataLayout::kNCHW &&
          data_layout != framework::DataLayout::kNHWC)
        return false;
      auto interp_method =
          BOOST_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "nearest") return false;
    }

    if (op_type == "roi_align") {
      if (!with_dynamic_shape) return false;

      std::vector<std::string> attrs{"pooled_height", "pooled_width",
                                     "spatial_scale", "sampling_ratio"};
      for (auto const attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }

      const auto pooled_height =
          BOOST_GET_CONST(int, desc.GetAttr("pooled_height"));
      if (pooled_height <= 0) return false;

      const auto pooled_width =
          BOOST_GET_CONST(int, desc.GetAttr("pooled_width"));
      if (pooled_width <= 0) return false;

      const auto spatial_scale =
          BOOST_GET_CONST(float, desc.GetAttr("spatial_scale"));
      if (spatial_scale <= 0.f) return false;
    }

    if (op_type == "hard_swish") {
      if (desc.Input("X").size() != 1) {
        VLOG(1) << "HardSwish op has only 1 input, but got "
                << desc.Input("X").size();
        return false;
      }

      if (desc.Output("Out").size() != 1) {
        VLOG(1) << "HardSwish op has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }
    }

    if (op_type == "batch_norm") {
      const std::vector<std::string> bn_inputs = {"X", "Bias", "Mean", "Scale",
                                                  "Variance"};
      auto* block = desc.Block();
      for (int i = 0; i < bn_inputs.size(); i++) {
        if (desc.Input(bn_inputs[i]).size() != 1) {
          VLOG(1) << "Invalid " << bn_inputs[i]
                  << "'s size of batch_norm TRT "
                     "converter. Expected 1, received "
                  << desc.Input(bn_inputs[i]).size() << ".";
          return false;
        }

        // "Bias", "Mean", "Scale", "Variance"
        if (i != 0) {
          auto* var_desc = block->FindVar(desc.Input(bn_inputs[i]).front());
          if (var_desc == nullptr) {
            VLOG(1) << "Variable of " << bn_inputs[i]
                    << " of batch_norm TRT converter is not found.";
            return false;
          }
        }
      }

      if (desc.Output("Y").size() != 1) {
        VLOG(1) << "Invalid output Y's size of batch_norm TRT "
                   "converter. Expected 1, received "
                << desc.Output("Y").size() << ".";
        return false;
      }
    }

    if (op_type == "split") {
      if (desc.Input("X").size() != 1) {
        VLOG(1) << "Invalid input X's size of split TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
        int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
        if (axis == 0) {
          VLOG(1) << "Invalid split axis. Split on batch is not supported in "
                     "TensorRT";
          return false;
        }
      }
    }

    if (op_type == "slice") {
      if (!desc.HasAttr("axes") || !desc.HasAttr("starts") ||
          !desc.HasAttr("ends")) {
        return false;
      } else {
        std::vector<int> axes =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("axes"));
        std::vector<int> starts =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("starts"));
        std::vector<int> ends =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("ends"));
        if (axes.size() != starts.size() || axes.size() != ends.size()) {
          return false;
        }
        if (!with_dynamic_shape) {
          for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] == 0) {
              VLOG(1) << "Invalid slice axis. Slice on batch axis is not "
                         "supported in TensorRT";
              return false;
            }
          }
        }
      }
    }

    if (op_type == "elementwise_add" || op_type == "elementwise_mul") {
      if (desc.Input("X").size() != 1) {
        VLOG(1) << "The input op's Input(\"X\").size() "
                   "should equal to 1, but received Input(\"X\").size() = "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Input("Y").size() != 1) {
        VLOG(1) << "The input op's Input(\"Y\").size() "
                   "should equal to 1, but received Input(\"Y\").size() = "
                << desc.Input("Y").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(1) << "The input op's Output(\"Out\").size() "
                   "should equal to 1, but reveceid Output(\"Out\").size() = "
                << desc.Output("Out").size() << ".";
        return false;
      }
    }

#if IS_TRT_VERSION_GE(6000)
    if (op_type == "stack") {
      if (!with_dynamic_shape) {
        VLOG(1)
            << "You are running the Ernie(Bert) model in static"
               "shape mode, which is not supported for the time being.\n"
               "You can use the config.SetTRTDynamicShapeInfo(...) interface"
               " to set the shape information to run the dynamic shape "
               "mode.";
        return false;
      }
    }
#endif

    if ((*teller)(op_type, desc, use_no_calib_int8)) return true;
  }
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

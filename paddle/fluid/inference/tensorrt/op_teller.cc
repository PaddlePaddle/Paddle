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
#include <bitset>
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
    int8_teller_set.insert("fused_embedding_eltwise_layernorm");
    int8_teller_set.insert("multihead_matmul");
    int8_teller_set.insert("skip_layernorm");
    int8_teller_set.insert("slice");
#endif
// TODO(baoachun) The group_norm trt plugin will check input's dim
// not -1 failed when dynamic shape mode.
// #if IS_TRT_VERSION_GE(7130)
//     teller_set.insert("group_norm");
// #endif
#if IS_TRT_VERSION_GE(7000)
    teller_set.insert("tile");
    teller_set.insert("flatten_contiguous_range");
#endif
#if CUDA_VERSION >= 10020
    teller_set.insert("reshape");
    teller_set.insert("reshape2");
    int8_teller_set.insert("reshape");
    int8_teller_set.insert("reshape2");
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
                                                  "depthwise_conv2d_transpose",
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
                                                  "gather_nd",
                                                  "yolo_box",
                                                  "roi_align",
                                                  "affine_channel",
                                                  "nearest_interp",
                                                  "anchor_generator",
                                                  "reduce_sum",
                                                  "reduce_mean",
                                                  "conv3d",
                                                  "conv3d_transpose",
                                                  "mish",
                                                  "nearest_interp_v2",
                                                  "pool3d",
                                                  "deformable_conv"};
  std::unordered_set<std::string> teller_set{"mul",
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
                                             "depthwise_conv2d_transpose",
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
                                             "gather_nd",
                                             "yolo_box",
                                             "roi_align",
                                             "affine_channel",
                                             "nearest_interp",
                                             "anchor_generator",
                                             "reduce_sum",
                                             "reduce_mean",
                                             "conv3d",
                                             "conv3d_transpose",
                                             "mish",
                                             "nearest_interp_v2",
                                             "pool3d",
                                             "deformable_conv"};
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
    if (op_type == "relu" || op_type == "relu6" || op_type == "tanh" ||
        op_type == "sigmoid") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << op_type
                << " op does not support input's dim is 1 in tensorrt.";
        return false;
      }
    }

    if (op_type == "pool2d") {
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      if (paddings.size() > 2) {
        return false;
      }
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "TRT Pool2d expect 1 input, but got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "TRT Pool2d has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }
      if (!desc.HasAttr("pooling_type")) {
        return false;
      } else {
        std::string pool_type =
            BOOST_GET_CONST(std::string, desc.GetAttr("pooling_type"));
        if (pool_type != "max" && pool_type != "avg") {
          VLOG(3) << "Wrong pool op type, the trt do not support the "
                  << pool_type << " pool type.";
          return false;
        }
        if (pool_type == "avg") {
          if (desc.HasAttr("global_pooling")) {
            if (!BOOST_GET_CONST(bool, desc.GetAttr("global_pooling"))) {
              if (desc.HasAttr("exclusive")) {
                if (BOOST_GET_CONST(bool, desc.GetAttr("exclusive"))) {
                  std::vector<int> ksize =
                      BOOST_GET_CONST(std::vector<int>, desc.GetAttr("ksize"));
                  for (size_t i = 0; i < ksize.size(); i++) {
                    if (ksize[i] <= paddings[i]) {
                      VLOG(3) << "the padding size should be less than the "
                                 "filter size "
                                 "for exclusive-counting pooling.";
                      return false;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    if (op_type == "conv2d" || op_type == "conv2d_transpose" ||
        op_type == "conv2d_fusion" || op_type == "depthwise_conv2d" ||
        op_type == "depthwise_conv2d_transpose") {
      if (desc.Input("Input").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 input, but got "
                << desc.Input("Input").size() << " input.";
        return false;
      }

      if (desc.Input("Filter").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 filter, but got "
                << desc.Input("Filter").size() << " filter.";
        return false;
      }

      if (desc.HasAttr("enable_int8")) {
        if (op_type == "conv2d" || op_type == "conv2d_fusion") {
          if (!desc.HasAttr("Input_scale")) {
            VLOG(3) << "Input scale not found. TRT int8"
                       " requires conv/deconv to have "
                       "input quantization scales.";
            return false;
          }
        }
      }

      if (op_type == "conv2d_transpose" ||
          op_type == "depthwise_conv2d_transpose") {
        if (!desc.HasAttr("dilations")) {
          return false;
        } else {
          const std::vector<int> dilations =
              BOOST_GET_CONST(std::vector<int>, desc.GetAttr("dilations"));
          if (dilations[0] != 1 || dilations[1] != 1) {
            VLOG(3) << "In conv2d_transpose, Dilations must be (1, 1) for "
                       "tensorRT, but given ("
                    << dilations[0] << ", " << dilations[1] << ")";
            return false;
          }
        }
      }

      if (desc.Output("Output").size() != 1) {
        VLOG(3) << "TRT Conv2d expect 1 output, but got "
                << desc.Output("Output").size() << " output.";
        return false;
      }

// strides > 1 and 'SAME' is only supported by trt7.0 above
#if !IS_TRT_VERSION_GE(7000)
      if (op_type == "conv2d" || op_type == "conv2d_fusion" ||
          op_type == "depthwise_conv2d") {
        if (desc.HasAttr("padding_algorithm") && with_dynamic_shape) {
          auto padding_algorithm =
              BOOST_GET_CONST(std::string, desc.GetAttr("padding_algorithm"));
          if (padding_algorithm == "SAME" && desc.HasAttr("strides")) {
            const std::vector<int> strides =
                BOOST_GET_CONST(std::vector<int>, desc.GetAttr("strides"));
            // there is no issue if strides.size() less than 2
            if (strides.size() > 1) {
              for (size_t i = 0; i < strides.size(); i++) {
                if (strides[i] > 1) return false;
              }
            }
          }
        }
      }
#endif
    }

    if (op_type == "deformable_conv") {
      if (with_dynamic_shape) {
        VLOG(3) << "Deformable conv trt plugin does not support dynamic shape";
        return false;
      }
      auto* block = desc.Block();
      auto input_name = desc.Input("Input")[0];
      auto* input_desc = block->FindVar(input_name);
      const auto input_shape = input_desc->GetShape();

      if (input_shape.size() != 4) {
        VLOG(3) << "Input of deformable conv should be 4-D Tensor, but got "
                << input_shape.size();
        return false;
      }

      auto filter_name = desc.Input("Filter")[0];
      auto* filter_desc = block->FindVar(filter_name);
      const auto filter_shape = filter_desc->GetShape();

      int groups = BOOST_GET_CONST(int, desc.GetAttr("groups"));
      if (input_shape[1] != filter_shape[1] * groups) {
        VLOG(3) << "The number of input channels should be equal to filter "
                << "channels * groups. But got input channels "
                << input_shape[1] << "filter channels " << filter_shape[1];
        return false;
      }

      const std::vector<int> strides =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("strides"));
      if (strides.size() != 2) {
        VLOG(3) << "The size of strides should be 2, but got "
                << strides.size();
        return false;
      }

      const std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      if (paddings.size() != 2) {
        VLOG(3) << "The size of paddings shoule be 2, but got "
                << paddings.size();
        return false;
      }
    }

    if (op_type == "matmul") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

      // not support broadcast
      auto* x_var_desc = block->FindVar(desc.Input("X")[0]);
      auto* y_var_desc = block->FindVar(desc.Input("Y")[0]);
      const auto x_shape = x_var_desc->GetShape();
      const auto y_shape = y_var_desc->GetShape();
      if (x_shape.size() != y_shape.size()) {
        VLOG(3)
            << "matmul op not support broadcast, please check inputs'shape. ";
        return false;
      }
      uint64_t dims = 2;
      for (size_t i = 0; i < x_shape.size() - dims; ++i) {
        if (x_shape[i] != y_shape[i] && (x_shape[i] == 1 || y_shape[i] == 1)) {
          VLOG(3) << "matmul op not support broadcast, please check "
                     "inputs'shape[i]. ";
          return false;
        }
      }

      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() < 3) {
            VLOG(3)
                << "matmul op dims < 3 not supported in tensorrt, but got dims "
                << shape.size() << ", so jump it.";
            return false;
          }
        }
      }
    }
    if (op_type == "softmax") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
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
      }
      int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
      if (with_dynamic_shape) {
        if (axis < 0) return false;
      } else {
        if (axis <= 0) return false;
      }
      auto concat_inputs = desc.Inputs();
      if (concat_inputs.find("AxisTensor") != concat_inputs.end()) {
        if (desc.Input("AxisTensor").size() >= 1) {
          return false;
        }
      }
    }
    if (op_type == "transpose2" || op_type == "transpose") {
      if (!desc.HasAttr("axis")) {
        return false;
      }
      std::vector<int> axis =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("axis"));
      if (!with_dynamic_shape && axis[0] != 0) return false;
      if (axis.size() >= nvinfer1::Dims::MAX_DIMS) return false;

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (axis.size() != x_shape.size()) return false;
      int dims = x_shape.size();

      std::vector<int> perm(nvinfer1::Dims::MAX_DIMS);
      for (int i = 0; i < dims; i++) {
        perm[i] = axis[i];
      }
      auto is_valid_permutation = [&](int dims,
                                      const std::vector<int>& permutation) {
        std::bitset<nvinfer1::Dims::MAX_DIMS> found;
        for (int i = 0; i < dims; ++i) {
          const int x = permutation[i];
          if ((x < 0) || (x >= dims) || found[x])
            return false;  // Out of bounds or duplicate
          found.set(x);
        }
        return true;
      };
      if (!is_valid_permutation(dims, perm)) {
        VLOG(3) << "Invalid permutation dimensions for trt transpose op "
                   "converter: duplicate or out of bound.";
        return false;
      }
    }
    if (op_type == "flatten2" || op_type == "flatten") {
      if (!desc.HasAttr("axis")) {
        return false;
      } else {
#if IS_TRT_VERSION_GE(7130)
#else
        if (with_dynamic_shape) return false;
#endif
        int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));
        if (axis != 1) return false;
      }
    }
    if (op_type == "flatten_contiguous_range") {
      if (!with_dynamic_shape) {
        int start_axis = BOOST_GET_CONST(int, desc.GetAttr("start_axis"));
        int stop_axis = BOOST_GET_CONST(int, desc.GetAttr("stop_axis"));
        auto x_var_name = desc.Input("X")[0];
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }
        auto* x_var_desc = block->FindVar(x_var_name);
        const auto x_shape = x_var_desc->GetShape();
        int dims = x_shape.size();
        if (start_axis < 0) start_axis += dims;
        if (start_axis == 0) {
          VLOG(3) << "TRT flatten_contiguous_range not support the "
                     "batch-dimension being changed";
          return false;
        }
        if (stop_axis < 0) stop_axis += dims;
        for (int i = start_axis; i <= stop_axis; ++i) {
          if (x_shape[i] < 0) {
            VLOG(3) << "On TRT static shape,flatten_contiguous_range input dim "
                       "should be > 0";
            return false;
          }
        }
      }
    }

    if (op_type == "gather") {
      auto gather_inputs = desc.Inputs();
      if (gather_inputs.find("Axis") != gather_inputs.end()) {
        if (desc.Input("Axis").size() >= 1) {
          return false;
        }
      }
      if (!with_dynamic_shape) {
        return false;
      } else {
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }
        auto* x_var_desc = block->FindVar(desc.Input("X")[0]);
        const auto x_shape = x_var_desc->GetShape();
        if (x_shape.size() == 1) {
          VLOG(3) << "Gather does not support 1-dimensional input in tensorrt";
          return false;
        }
      }
    }

    if (op_type == "gather_nd") {
      if (!with_dynamic_shape) return false;

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto index_var_name = desc.Input("Index")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      auto* index_var_desc = block->FindVar(index_var_name);

      // The index input must be int32 datatype.
      if (index_var_desc->GetDataType() !=
          paddle::framework::proto::VarType_Type::VarType_Type_INT32) {
        VLOG(3) << "gather_nd op Index input data type must be int32";
        return false;
      }

      const auto index_shape = index_var_desc->GetShape();
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() <= 2) {
        VLOG(3) << "gather_nd op requires the input's dimension to be greater "
                   "than 2";
        return false;
      }

      if (x_shape.size() != index_shape.size()) {
        VLOG(3) << "gather_nd op Index input dims size [" << index_shape.size()
                << " ] not equal to x dims size [" << x_shape.size() << "]";
        return false;
      }
    }

    if (op_type == "anchor_generator") {
      if (!with_dynamic_shape) return false;
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

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 2) {
        return false;
      }
    }

    if (op_type == "multiclass_nms") {
      if (with_dynamic_shape) return false;
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          const auto shape = var_desc->GetShape();
          if (shape.size() != 3) {
            VLOG(3) << "multiclass_nms op dims != 3 not supported in tensorrt, "
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

    if (op_type == "nearest_interp") {
      std::vector<std::string> attrs{"interp_method", "align_corners", "scale",
                                     "out_h", "out_w"};
      for (auto const attr : attrs) {
        if (!desc.HasAttr(attr)) return false;
      }
      if (desc.HasAttr("data_layout")) {
        auto data_layout = framework::StringToDataLayout(
            BOOST_GET_CONST(std::string, desc.GetAttr("data_layout")));
        if (data_layout != framework::DataLayout::kNCHW &&
            data_layout != framework::DataLayout::kNHWC)
          return false;
      }
      auto interp_method =
          BOOST_GET_CONST(std::string, desc.GetAttr("interp_method"));
      if (interp_method != "nearest") return false;
      auto scale = BOOST_GET_CONST(float, desc.GetAttr("scale"));
      auto out_h = BOOST_GET_CONST(int, desc.GetAttr("out_h"));
      auto out_w = BOOST_GET_CONST(int, desc.GetAttr("out_w"));
      auto align_corners = BOOST_GET_CONST(bool, desc.GetAttr("align_corners"));
      if (!(scale > 0.f && (out_h <= 0 && out_w <= 0))) {
        if (out_h <= 0) {
          VLOG(3) << "out_h must be greater than 0 if scale is not set.";
          return false;
        }
        if (out_w <= 0) {
          VLOG(3) << "out_w must be greater than 0 if scale is not set.";
          return false;
        }
      }
      if ((scale <= 0.f) && with_dynamic_shape) {
        VLOG(3) << "dynamic shape not support scale not set.";
        return false;
      }
      // When align_corners = true, the paddle's and trt_layer's results has
      // diff
      if (align_corners && scale != 1) {
        return false;
      }
    }

    if (op_type == "nearest_interp_v2") {
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
      auto scale = BOOST_GET_CONST(std::vector<float>, desc.GetAttr("scale"));
      auto out_h = BOOST_GET_CONST(int, desc.GetAttr("out_h"));
      auto out_w = BOOST_GET_CONST(int, desc.GetAttr("out_w"));
      if (!(out_h > 0 && out_w > 0)) {
        if (scale.size() < 2) return false;
        if (scale[0] <= 0.f || scale[1] <= 0.f) {
          VLOG(3) << "scale factor must be greater than 0 if out_h or out_w is "
                     "not set.";
          return false;
        }
      }
    }

    if (op_type == "hard_swish") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "HardSwish op has only 1 input, but got "
                << desc.Input("X").size();
        return false;
      }

      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "HardSwish op has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }
    }

    if (op_type == "batch_norm") {
      const std::vector<std::string> bn_inputs = {"X", "Bias", "Mean", "Scale",
                                                  "Variance"};
      for (unsigned int i = 0; i < bn_inputs.size(); i++) {
        if (desc.Input(bn_inputs[i]).size() != 1) {
          VLOG(3) << "Invalid " << bn_inputs[i]
                  << "'s size of batch_norm TRT "
                     "converter. Expected 1, received "
                  << desc.Input(bn_inputs[i]).size() << ".";
          return false;
        }
      }
      auto batch_norm_inputs = desc.Inputs();
      if (batch_norm_inputs.find("MomentumTensor") != batch_norm_inputs.end()) {
        if (desc.Input("MomentumTensor").size() >= 1) {
          return false;
        }
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "Invalid output Y's size of batch_norm TRT "
                   "converter. Expected 1, received "
                << desc.Output("Y").size() << ".";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
    }

    if (op_type == "split") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of split TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      auto split_inputs = desc.Inputs();
      if (split_inputs.find("AxisTensor") != split_inputs.end()) {
        if (desc.Input("AxisTensor").size() >= 1) {
          return false;
        }
      }
      if (split_inputs.find("SectionsTensorList") != split_inputs.end()) {
        if (desc.Input("SectionsTensorList").size() >= 1) {
          return false;
        }
      }
      if (!desc.HasAttr("axis")) {
        return false;
      }
      int axis = BOOST_GET_CONST(int, desc.GetAttr("axis"));

      if (axis == 0) {
        VLOG(3) << "Invalid split axis. Split on batch is not supported in "
                   "TensorRT";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      size_t output_num = desc.Output("Out").size();
      std::vector<int> output_lengths;
      int num = 0;
      if (desc.HasAttr("num")) {
        num = BOOST_GET_CONST(int, desc.GetAttr("num"));
      }
      if (desc.HasAttr("sections")) {
        output_lengths =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("sections"));
      }
      if (output_lengths.size() == 0 && num == 0) {
        VLOG(3) << "sections and num cannot be equal to 0 at the same time";
        return false;
      }
      if (with_dynamic_shape) {
#if IS_TRT_VERSION_GE(6000)
#else
        VLOG(3) << "You are running the TRT Dynamic Shape mode, need to "
                   "confirm that "
                   "your TRT version is no less than 6.0";
        return false;
#endif
      }
      axis += (axis < 0) ? x_shape.size() : 0;
      if (x_shape[axis] == -1) {
        VLOG(3) << "The (" << axis << ") dim of input should not be -1";
        return false;
      }
      if (output_lengths.size() == 0) {
        if (num > 0) {
          int64_t in_axis_dim = x_shape[axis];
          if (in_axis_dim % num != 0) {
            VLOG(3) << "Invalid number to split. Tensor split does not result"
                       " in an equal division of dimensions. Axis dim = "
                    << in_axis_dim << " num = " << num << "!= 0";
            return false;
          }
          size_t out_axis_dim = in_axis_dim / num;
          for (int i = 0; i < num; ++i) {
            output_lengths.push_back(out_axis_dim);
          }
        }
      }
      if (output_lengths.size() != output_num) {
        VLOG(3) << "The output_length should be equal to the output size.";
        return false;
      }
    }

    if (op_type == "scale") {
      auto scale_inputs = desc.Inputs();
      if (scale_inputs.find("ScaleTensor") != scale_inputs.end()) {
        if (desc.Input("ScaleTensor").size() >= 1) {
          return false;
        }
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (!with_dynamic_shape && x_shape.size() == 1) {
        VLOG(3) << "Scale op does not support 1-dimensional input in tensorrt";
        return false;
      }
    }

    if (op_type == "slice") {
      if (desc.HasAttr("decrease_axis")) {
        std::vector<int> decrease_axis =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("decrease_axis"));
        if (decrease_axis.size() > 0) {
          VLOG(3) << "Invalid slice decrease_axis. decrease_axis.size() > 0"
                     "is not supported in TensorRT";
          return false;
        }
      }

      if (!desc.HasAttr("axes") || !desc.HasAttr("starts") ||
          !desc.HasAttr("ends")) {
        VLOG(3) << "The necessary attributes of the slice operator axes "
                   "or starts or ends are missing.";
        return false;
      } else {
        std::vector<int> axes =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("axes"));
        std::vector<int> starts =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("starts"));
        std::vector<int> ends =
            BOOST_GET_CONST(std::vector<int>, desc.GetAttr("ends"));

        if (axes.size() != starts.size() || axes.size() != ends.size()) {
          VLOG(3) << "The shape of attributes of the slice operator axes "
                     "or starts or ends are not equal.";
          return false;
        }
        if (!with_dynamic_shape) {
          for (size_t i = 0; i < axes.size(); i++) {
            if (axes[i] == 0) {
              VLOG(3) << "Invalid slice axis. Slice on batch axis is not "
                         "supported in TensorRT";
              return false;
            }
          }
        } else {
          for (size_t i = 0; i < axes.size(); i++) {
            if (starts[i] < 0 || ends[i] < 0) {
              VLOG(3) << "Invalid slice attribute 'starts' or 'ends'. "
                         "Negative starts or ends not supported in TensorRT "
                         "when running in dynamic shape mode.";
              return false;
            }
          }
        }
      }
    }

    if (op_type == "elementwise_add" || op_type == "elementwise_mul") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "The input op's Input(\"X\").size() "
                   "should equal to 1, but received Input(\"X\").size() = "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Input("Y").size() != 1) {
        VLOG(3) << "The input op's Input(\"Y\").size() "
                   "should equal to 1, but received Input(\"Y\").size() = "
                << desc.Input("Y").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "The input op's Output(\"Out\").size() "
                   "should equal to 1, but reveceid Output(\"Out\").size() = "
                << desc.Output("Out").size() << ".";
        return false;
      }
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* x_var_desc = block->FindVar(desc.Input("X")[0]);
      auto* y_var_desc = block->FindVar(desc.Input("Y")[0]);
      const auto x_shape = x_var_desc->GetShape();
      const auto y_shape = y_var_desc->GetShape();
      if (x_shape.size() == 1 && y_shape.size() == 1) {
        VLOG(3) << "Now trt may not support two 1d tensor elementwise op.";
        return false;
      }
    }

    if (op_type == "stack") {
      if (!with_dynamic_shape) {
        VLOG(3)
            << "static shape mode is not supported for TRT stack.\n"
               "You can use the config.SetTRTDynamicShapeInfo(...) interface"
               " to set the shape information to run the dynamic shape "
               "mode.";
        return false;
      }
    }

    if (op_type == "fused_embedding_eltwise_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "fused_embedding_eltwise_layernorm should run on dynamic "
                   "shape mode.";
        return false;
      }
      if (desc.Input("Ids").size() != desc.Input("Embs").size()) {
        VLOG(3) << "The id and emb size of fused EmbEltwiseLayerNormOp "
                   "should be same ";
        return false;
      }
    }

    if (op_type == "gelu") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "gelu op has only 1 input, but got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "gelu op has only 1 output, but got "
                << desc.Output("Out").size();
        return false;
      }

      if (desc.HasAttr("approximate")) {
        if (BOOST_GET_CONST(bool, desc.GetAttr("approximate"))) return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << "gelu op does not support input's dim is 1 in tensorrt.";
        return false;
      }
    }

    if (op_type == "layer_norm") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "input of layer_norm op converter should be 1, got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Input("Bias").size() != 1) {
        VLOG(3) << "Bias of layer_norm op converter should be 1, got "
                << desc.Input("Bias").size();
        return false;
      }
      if (desc.Input("Scale").size() != 1) {
        VLOG(3) << "Scale of layer_norm op converter should be 1, got "
                << desc.Input("Scale").size();
        return false;
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "output of layer_norm op converter should be 1, got "
                << desc.Output("Y").size();
        return false;
      }
    }

    if (op_type == "instance_norm") {
      if (with_dynamic_shape) {
        VLOG(3) << "trt instance_norm op does not support dynamic shape ";
        return false;
      }
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "input of instance_norm op converter should be 1, got "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Input("Bias").size() != 1) {
        VLOG(3) << "Bias of instance_norm op converter should be 1, got "
                << desc.Input("Bias").size();
        return false;
      }
      if (desc.Input("Scale").size() != 1) {
        VLOG(3) << "Scale of instance_norm op converter should be 1, got "
                << desc.Input("Scale").size();
        return false;
      }
      if (desc.Output("Y").size() != 1) {
        VLOG(3) << "output of layer_norm op converter should be 1, got "
                << desc.Output("Y").size();
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() != 4) {
        VLOG(3) << "The instance_norm op only support 4-dimensional input in "
                   "tensorrt.";
        return false;
      }
    }

    if (op_type == "leaky_relu") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid number of TRT leaky_relu op converter "
                   "inputs. Expected 1, but received "
                << desc.Input("X").size();
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "output of leaky_relu op converter should be 1, got "
                << desc.Output("Out").size();
        return false;
      }
    }

    if (op_type == "pad") {
      const float pad_value = BOOST_GET_CONST(float, desc.GetAttr("pad_value"));
      if (pad_value != 0.0f) {
        VLOG(3) << "The pad layer of TRT only support zero.";
        return false;
      }
      std::vector<int64_t> shape;
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      for (auto& param_name : desc.Inputs()) {
        for (auto& var_name : param_name.second) {
          auto* var_desc = block->FindVar(var_name);
          shape = var_desc->GetShape();
        }
      }
      int nbDims = shape.size();
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));
      int pad_size = paddings.size();
      if (nbDims < 2) {
        return false;
      }
      if (nbDims * 2 != pad_size) {
        return false;
      }
      for (int i = 0; i < pad_size - 4; i++) {
        if (paddings[i] != 0) {
          return false;
        }
      }
    }

    if (op_type == "swish") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << "swish op does not support input's dim is 1 in tensorrt.";
        return false;
      }
    }

    if (op_type == "prelu") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of prelu TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "Invalid output Out's size of prelu TRT converter. "
                   "Expected 1, received "
                << desc.Output("Out").size() << ".";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* var_desc = block->FindVar(desc.Input("Alpha")[0]);
      if (!var_desc) {
        VLOG(3) << "Variable Alpha of prelu TRT converter not found.";
        return false;
      }

      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << "prelu op does not support input's dim is 1 in tensorrt.";
        return false;
      }

#if IS_TRT_VERSION_LT(7000)
      if (!with_dynamic_shape) {
        // TODO(inference): fix trt6 static plugin error.
        VLOG(3) << "prelu static plugin in trt6 has bug.";
        return false;
      }
#endif
    }

    if (op_type == "mish") {
      if (desc.Input("X").size() != 1) {
        VLOG(3) << "Invalid input X's size of mish TRT converter. "
                   "Expected 1, received "
                << desc.Input("X").size() << ".";
        return false;
      }
      if (desc.Output("Out").size() != 1) {
        VLOG(3) << "Invalid output Out's size of mish TRT converter. "
                   "Expected 1, received "
                << desc.Output("Out").size() << ".";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << "mish op does not support input's dim is 1 in tensorrt.";
        return false;
      }

      if (!with_dynamic_shape) {
        if (x_shape.size() == 2) {
          VLOG(3) << "mish op does not support input's dim is 2 in tensorrt.";
          return false;
        }
      }
    }

    if (op_type == "roi_align") {
      if (!with_dynamic_shape) {
        VLOG(3) << "TRT roi align plugin only accept the dynamic shape, "
                   "because that "
                   "the roi_align will change the batch size.";
        return false;
      }
      std::vector<std::string> attrs{"pooled_height", "pooled_width",
                                     "spatial_scale", "sampling_ratio",
                                     "aligned"};
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

      auto roi_align_inputs = desc.Inputs();
      if (roi_align_inputs.find("RoisNum") != roi_align_inputs.end()) {
        if (desc.Input("RoisNum").size() >= 1) {
          return false;
        }
      }
    }

    if (op_type == "shuffle_channel") {
      if (with_dynamic_shape) {
        VLOG(3) << "You are running the TRT Dynamic Shape mode, "
                   "the shuffle_channel op does not support dynamic shape yet";
        return false;
      }
    }

    if (op_type == "skip_layernorm") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the skip_layernorm does not support static shape yet";
        return false;
      }
    }

    if (op_type == "multihead_matmul") {
      if (!with_dynamic_shape) {
        VLOG(3) << "the multihead_matmul does not support static shape yet";
        return false;
      }

      if (desc.HasAttr("enable_int8") && !desc.HasAttr("Input_scale")) {
        VLOG(3) << "Multihead layers must have input scale in int8 mode.";
        return false;
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto* input_desc = block->FindVar(desc.Input("Input").front());
      const auto input_shape = input_desc->GetShape();
      const auto head_number =
          BOOST_GET_CONST(int, desc.GetAttr("head_number"));

      auto* biasqk_desc = block->FindVar(desc.Input("BiasQK").front());
      const auto biasqk_shape = biasqk_desc->GetShape();
      // The BiasQK's shape requires to be
      // [batch, 1, 1, length] or [batch, head, length, length].
      bool has_same_shape = head_number == biasqk_shape[1] &&
                            input_shape[1] == biasqk_shape[2] &&
                            input_shape[1] == biasqk_shape[3];
      bool is_broadcastable = biasqk_shape[1] == 1 && biasqk_shape[2] == 1 &&
                              input_shape[1] == biasqk_shape[3];
      if (!(has_same_shape || is_broadcastable)) {
        VLOG(3) << "The BiasQK's shape is invalid, expect [" << input_shape[0]
                << ", 1, 1, " << input_shape[1] << "] or [" << input_shape[0]
                << ", " << head_number << ", " << input_shape[1] << ", "
                << input_shape[1] << "] but [" << biasqk_shape[0] << ", "
                << biasqk_shape[1] << ", " << biasqk_shape[2] << ", "
                << biasqk_shape[3] << "].";
        return false;
      }
    }

    if (op_type == "fc") {
      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }

      // y'shapes == 2
      auto fc_inputs = desc.Inputs();
      std::string fc_y = "";
      if (fc_inputs.find("Y") != fc_inputs.end()) {
        fc_y = "Y";
      } else if (fc_inputs.find("W") != fc_inputs.end()) {
        fc_y = "W";
      } else {
        VLOG(3) << " input_y(fc_op) must be Y or W ";
        return false;
      }

      //  There is currently no input: Y(weight) more than two dimensions
      /*
      auto* y_var_desc = block->FindVar(desc.Input(fc_y)[0]);
      const auto y_shape = y_var_desc->GetShape();
      if (y_shape.size() != 2) {
        VLOG(3)
            << " input_y(fc_op)'shapes must be 2, but input_y(fc_op)'shapes = "
            << y_shape.size();
        return false;
      }
      // y_num_col_dims ==1
      if (desc.HasAttr("y_num_col_dims")) {
        int y_num_col_dims =
            BOOST_GET_CONST(int, desc.GetAttr("y_num_col_dims"));
        if (y_num_col_dims != 1) {
          VLOG(3) << " fc_op'y_num_col_dims must be 1, but y_num_col_dims = "
                  << y_num_col_dims;
          return false;
        }
      }
      */
      int x_num_col_dims =
          desc.HasAttr("x_num_col_dims")
              ? BOOST_GET_CONST(int, desc.GetAttr("x_num_col_dims"))
              : (desc.HasAttr("in_num_col_dims")
                     ? BOOST_GET_CONST(int, desc.GetAttr("in_num_col_dims"))
                     : 1);
      if (x_num_col_dims < 1) {
        VLOG(3) << "fc_op expects x_num_col_dims >= 1, "
                   "but x_num_col_dims = "
                << x_num_col_dims;
        return false;
      }
    }

    if (op_type == "reshape" || op_type == "reshape2") {
      if (!desc.HasAttr("shape")) {
        return false;
      }
      // Paddle-TRT does not support the input tensors: Shape and ShapeTensor
      auto reshape_inputs = desc.Inputs();
      if (reshape_inputs.find("Shape") != reshape_inputs.end()) {
        if (desc.Input("Shape").size() >= 1) {
          return false;
        }
      }
      if (reshape_inputs.find("ShapeTensor") != reshape_inputs.end()) {
        if (desc.Input("ShapeTensor").size() >= 1) {
          return false;
        }
      }
      std::vector<int> shape =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("shape"));
      if (shape.size() >= nvinfer1::Dims::MAX_DIMS) return false;
      if (!with_dynamic_shape && (shape[0] == -1 || shape.size() == 1))
        return false;
    }

    if (op_type == "clip") {
      // Paddle-TRT does not support the input tensors: Min and Max
      auto clip_inputs = desc.Inputs();
      if (clip_inputs.find("Min") != clip_inputs.end()) {
        if (desc.Input("Min").size() >= 1) {
          return false;
        }
      }
      if (clip_inputs.find("Max") != clip_inputs.end()) {
        if (desc.Input("Max").size() >= 1) {
          return false;
        }
      }

      auto* block = desc.Block();
      if (block == nullptr) {
        VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                   "Developers need to check whether block_desc is passed in "
                   "the pass.";
        return false;
      }
      auto x_var_name = desc.Input("X")[0];
      auto* x_var_desc = block->FindVar(x_var_name);
      const auto x_shape = x_var_desc->GetShape();
      if (x_shape.size() == 1) {
        VLOG(3) << "clip op does not support input's dim is 1 in tensorrt.";
        return false;
      }
    }

    if (op_type == "reduce_sum" || op_type == "reduce_mean") {
      if (!(desc.HasAttr("keep_dim") && desc.HasAttr("dim") &&
            desc.HasAttr("reduce_all"))) {
        VLOG(3) << "the " << op_type
                << " does not have attr (keep_dim or dim or "
                   "reduce_all)";
        std::cout << "attr " << desc.HasAttr("keep_dim") << " "
                  << desc.HasAttr("dim") << " " << desc.HasAttr("reduce_all");
        return false;
      }

      // The batch size dimension cannot be reduced if it's not dynamic shape.
      if (!with_dynamic_shape) {
        if (BOOST_GET_CONST(bool, desc.GetAttr("reduce_all"))) return false;
        std::vector<int32_t> dim =
            BOOST_GET_CONST(std::vector<int32_t>, desc.GetAttr("dim"));
        for (auto x : dim) {
          if (!x) return false;
        }
      } else {
        if (BOOST_GET_CONST(bool, desc.GetAttr("reduce_all")) &&
            !BOOST_GET_CONST(bool, desc.GetAttr("keep_dim")))
          return false;
      }
      if (desc.HasAttr("out_dtype")) {
        int out_dtype = BOOST_GET_CONST(int32_t, desc.GetAttr("out_dtype"));
        if (out_dtype != -1) {
          return false;
        }
      }
    }
#if IS_TRT_VERSION_GE(7000)
    if (op_type == "tile") {
      // Paddle-TRT does not support the input tensors.
      auto tile_inputs = desc.Inputs();
      if (tile_inputs.find("repeat_times_tensor") != tile_inputs.end()) {
        if (desc.Input("repeat_times_tensor").size() >= 1) {
          return false;
        }
      }
      if (tile_inputs.find("RepeatTimes") != tile_inputs.end()) {
        if (desc.Input("RepeatTimes").size() >= 1) {
          return false;
        }
      }
      if (with_dynamic_shape) return false;
      if (!with_dynamic_shape && !desc.HasAttr("repeat_times")) return false;
    }
#endif

    if (op_type == "conv3d" || op_type == "conv3d_transpose") {
      if (desc.HasAttr("padding_algorithm")) {
        std::string padding_algorithm =
            BOOST_GET_CONST(std::string, desc.GetAttr("padding_algorithm"));

        // trt error is arised if conv3d_transpose and SAME
        if (op_type == "conv3d_transpose" && padding_algorithm == "SAME" &&
            !with_dynamic_shape) {
          return false;
        }
      }

#if !IS_TRT_VERSION_GE(7000)
      // looks like some issues with trt6.0
      if (with_dynamic_shape) {
        return false;
      }
#endif
      std::vector<int> paddings =
          BOOST_GET_CONST(std::vector<int>, desc.GetAttr("paddings"));

      // conv3d and conv3d_transpose need padding check
      if (paddings.size() > 3) return false;

      if (desc.Input("Input").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 input, but got "
                << desc.Input("Input").size() << " input.";
        return false;
      }

      if (desc.Input("Filter").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 filter, but got "
                << desc.Input("Filter").size() << " filter.";
        return false;
      }

      if (op_type == "conv3d_transpose") {
        if (!desc.HasAttr("dilations")) {
          return false;
        } else {
          const std::vector<int> dilations =
              BOOST_GET_CONST(std::vector<int>, desc.GetAttr("dilations"));
          if (dilations[0] != 1 || dilations[1] != 1 || dilations[2] != 1) {
            VLOG(3) << "In conv3d_transpose, Dilations must be (1, 1, 1) for "
                       "tensorRT, but given ("
                    << dilations[0] << ", " << dilations[1] << ", "
                    << dilations[2] << ")";
            return false;
          }
        }
      }

      if (desc.Output("Output").size() != 1) {
        VLOG(3) << "TRT Conv3d expect 1 output, but got "
                << desc.Output("Output").size() << " output.";
        return false;
      }
    }

    if (op_type == "hard_sigmoid") {
      if (!with_dynamic_shape) {
        auto* block = desc.Block();
        if (block == nullptr) {
          VLOG(3) << "The block desc is nullptr, we can't continue to analyze. "
                     "Developers need to check whether block_desc is passed in "
                     "the pass.";
          return false;
        }
        auto x_var_name = desc.Input("X")[0];
        auto* x_var_desc = block->FindVar(x_var_name);
        const auto x_shape = x_var_desc->GetShape();
        if (x_shape.size() == 1) {
          VLOG(3) << "Hard sigmoid does not support 1-dimensional input in "
                     "tensorrt";
          return false;
        }
      }
    }

    if ((*teller)(op_type, desc, use_no_calib_int8)) return true;
  }

  VLOG(3) << "trt unsupported op " << op_type;
  return false;
}

OpTeller::OpTeller() { tellers_.emplace_back(new SimpleOpTypeSetTeller); }

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

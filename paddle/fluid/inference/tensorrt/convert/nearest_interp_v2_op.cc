/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;
namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

class NearestInterpolateV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid nearest_interp_v2 op";
    VLOG(1) << "@@@convert a fluid nearest_interp_v2 op";
    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    auto data_layout = phi::StringToDataLayout(
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
    auto interp_method =
        PADDLE_GET_CONST(std::string, op_desc.GetAttr("interp_method"));
    bool align_corners =
        PADDLE_GET_CONST(bool, op_desc.GetAttr("align_corners"));
    VLOG(1) << "@@@ before resize_inputs = op_desc.Inputs";
    auto resize_inputs = op_desc.Inputs();
    VLOG(1) << "@@@ get const scale ";
    bool has_scale_input_attr =
        (resize_inputs.find("Scale") != resize_inputs.end());

    auto scale = PADDLE_GET_CONST(std::vector<float>, op_desc.GetAttr("scale"));
    VLOG(1) << "@@@ after get const scale ";

    auto out_h = PADDLE_GET_CONST(int, op_desc.GetAttr("out_h"));
    auto out_w = PADDLE_GET_CONST(int, op_desc.GetAttr("out_w"));

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    layer->setAlignCorners(align_corners);

    auto in_dim = input->getDimensions();

    float scale_h = -1.f;
    float scale_w = -1.f;

    std::vector<float> scales;

    if (out_h > 0 && out_w > 0) {
      // axis are different in static/dynamic mode
      bool with_dynamic = engine_->with_dynamic_shape();
      int h_axis = (data_layout == phi::DataLayout::kNCHW) + with_dynamic;
      int w_axis = (data_layout == phi::DataLayout::kNCHW) + 1 + with_dynamic;

      scale_h =
          static_cast<float>(out_h) / static_cast<float>(in_dim.d[h_axis]);
      scale_w =
          static_cast<float>(out_w) / static_cast<float>(in_dim.d[w_axis]);
    } else if (has_scale_input_attr && scale.size() > 1) {
      scale_h = scale[0];
      scale_w = scale[1];
    }
    nvinfer1::ITensor* outsize_tensor = nullptr;
    std::vector<nvinfer1::ITensor*> concat_inputs;
    VLOG(1) << "@@@ before get size tensor";
    if (engine_->with_dynamic_shape() &&
        resize_inputs.find("SizeTensor") != resize_inputs.end()) {
      if (op_desc.Input("SizeTensor").size() > 0) {
        VLOG(1) << "@@@ have size tensor, before concat";
        for (auto name : op_desc.Input("SizeTensor")) {
          concat_inputs.push_back(engine_->GetITensor(name));
          VLOG(1) << "@@@@ size tensor name: " << name;
        }
        VLOG(1) << "@@@ have size tensor, after concat";
        outsize_tensor = Concat(concat_inputs);
      }
    }

    if (engine_->with_dynamic_shape()) {
      scales.push_back(1.f);
    }

    if (data_layout == phi::DataLayout::kNCHW) {
      scales.push_back(1.f);
      scales.push_back(scale_h);
      scales.push_back(scale_w);
    } else if (data_layout == phi::DataLayout::kNHWC) {
      // NHWC
      scales.push_back(scale_h);
      scales.push_back(scale_w);
      scales.push_back(1.f);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Data layout must be NCHW or NHWC."));
    }

    if (engine_->with_dynamic_shape()) {
      if (outsize_tensor != nullptr) {
        std::vector<nvinfer1::ITensor*> outsize_itensors;
        auto* input_shape = Shape(input);
        outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 0));

        if (data_layout == phi::DataLayout::kNCHW) {
          outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 1));
          outsize_itensors.push_back(outsize_tensor);
        } else if (data_layout == phi::DataLayout::kNHWC) {
          outsize_itensors.push_back(outsize_tensor);
          outsize_itensors.push_back(GetEleTensorOfShape(input_shape, 3));
        }
        layer->setInput(1, *Concat(outsize_itensors));
      } else {
        layer->setScales(scales.data(), scales.size());
      }
    } else {
      layer->setScales(scales.data(), scales.size());
    }

    RreplenishLayerAndOutput(
        layer, "nearest_interp_v2", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(nearest_interp_v2, NearestInterpolateV2OpConverter);

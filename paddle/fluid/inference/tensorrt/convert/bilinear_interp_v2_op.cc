/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

class BilinearInterpolateV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid bilinear_interp_v2 op";

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    auto data_layout = framework::StringToDataLayout(
        BOOST_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
    auto interp_method =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("interp_method"));
    bool align_corners =
        BOOST_GET_CONST(bool, op_desc.GetAttr("align_corners"));
    auto align_mode = BOOST_GET_CONST(int, op_desc.GetAttr("align_mode"));

    auto resize_inputs = op_desc.Inputs();
    // A 1D int32 tensor
    nvinfer1::ITensor* outsize_tensor = nullptr;
    if (resize_inputs.find("OutSize") != resize_inputs.end()) {
      if (op_desc.Input("OutSize").size() >= 1) {
        outsize_tensor = engine_->GetITensor(op_desc.Input("OutSize")[0]);
      }
    }

    std::cout << outsize_tensor->getDimensions().d[0] << std::endl;
    std::cout << outsize_tensor->getDimensions().d[1] << std::endl;

    auto input_names = op_desc.Input("X");
    auto out_h = BOOST_GET_CONST(int, op_desc.GetAttr("out_h"));
    auto out_w = BOOST_GET_CONST(int, op_desc.GetAttr("out_w"));

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    if (align_mode == 0 && !align_corners) {
      layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
    }

    auto in_dim = input->getDimensions();
    float scale_h = -1.f;
    float scale_w = -1.f;

    // Scales Priority: Scale(tensor) > scale(attr) > out_d/out_h/out_w(attr)
    bool has_scale_input_attr =
        (resize_inputs.find("Scale") != resize_inputs.end());
    bool has_scale_input =
        has_scale_input_attr && (op_desc.Input("Scale").size() > 0);
    if (has_scale_input) {
      auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
      auto* scale_tensor = scale_var->GetMutable<framework::LoDTensor>();
      auto* scale_d = scale_tensor->data<float>();
      scale_h = scale_d[0];
      scale_w = scale_d[1];
    } else {
      const std::vector<float> scale_attr =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("scale"));
      if (scale_attr.size() > 1) {
        scale_h = scale_attr[0];
        scale_w = scale_attr[1];
      }
    }

    // axis are different in static/dynamic mode
    bool with_dynamic = engine_->with_dynamic_shape();
    int h_axis = (data_layout == framework::DataLayout::kNCHW) + with_dynamic;
    int w_axis =
        (data_layout == framework::DataLayout::kNCHW) + 1 + with_dynamic;

    if (scale_w > 0. && scale_h > 0.) {
      out_h = static_cast<int>(in_dim.d[h_axis] * scale_h);
      out_w = static_cast<int>(in_dim.d[w_axis] * scale_w);
    }

    if (out_h > 0 && out_w > 0) {
      scale_h =
          static_cast<float>(out_h) / static_cast<float>(in_dim.d[h_axis]);
      scale_w =
          static_cast<float>(out_w) / static_cast<float>(in_dim.d[w_axis]);
    }

    std::vector<float> scales;
    std::vector<nvinfer1::ITensor*> outsize_tensor_vec;
    nvinfer1::ITensor* input_shape_tensor = nullptr;

    if (engine_->with_dynamic_shape()) {
      input_shape_tensor = Shape(input);
      outsize_tensor_vec.push_back(GetEleTensorOfShape(input_shape_tensor, 0));
      scales.push_back(1.f);
    }

    if (data_layout == framework::DataLayout::kNCHW) {
      if(with_dynamic) {
        outsize_tensor_vec.push_back(GetEleTensorOfShape(input_shape_tensor, 1));
        outsize_tensor_vec.push_back(GetEleTensorOfShape(outsize_tensor, 0));
        outsize_tensor_vec.push_back(GetEleTensorOfShape(outsize_tensor, 1));
      }
      scales.push_back(1.f);
      scales.push_back(scale_h);
      scales.push_back(scale_w);
    } else if (data_layout == framework::DataLayout::kNHWC) {
      if(with_dynamic) {
        outsize_tensor_vec.push_back(GetEleTensorOfShape(outsize_tensor, 0));
        outsize_tensor_vec.push_back(GetEleTensorOfShape(outsize_tensor, 1));
        outsize_tensor_vec.push_back(GetEleTensorOfShape(input_shape_tensor, 3));
      }
      scales.push_back(scale_h);
      scales.push_back(scale_w);
      scales.push_back(1.f);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Data layout must be NCHW or NHWC."));
    }

    if (outsize_tensor) {
     // auto* tmp_tensor = Concat(outsize_tensor_vec);
     // std::vector<nvinfer1::ITensor*> tmp_vec{tmp_tensor, outsize_tensor};
      layer->setInput(1, *Concat(outsize_tensor_vec));
    } else {
      layer->setScales(scales.data(), scales.size());
    }

    std::cout << layer->getOutput(0)->getDimensions().d[0] << std::endl;
    std::cout << layer->getOutput(0)->getDimensions().d[1] << std::endl;
    std::cout << layer->getOutput(0)->getDimensions().d[2] << std::endl;
    std::cout << layer->getOutput(0)->getDimensions().d[3] << std::endl;

    RreplenishLayerAndOutput(
        layer, "bilinear_interp_v2", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(bilinear_interp_v2, BilinearInterpolateV2OpConverter);

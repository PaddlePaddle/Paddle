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

class BilinearInterpolateV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid bilinear_interp_v2 op";

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    auto data_layout = framework::StringToDataLayout(
        BOOST_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
    auto interp_method =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("interp_method"));

    auto input_names = op_desc.Input("X");
    auto out_h = BOOST_GET_CONST(int, op_desc.GetAttr("out_h"));
    auto out_w = BOOST_GET_CONST(int, op_desc.GetAttr("out_w"));
    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    layer->setAlignCorners(true);
    layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);
    auto in_dim = input->getDimensions();

    float scale_w = -1;
    float scale_h = -1;

    bool with_dynamic = engine_->with_dynamic_shape();
    int h_axis = (data_layout == framework::DataLayout::kNCHW) + with_dynamic;
    int w_axis =
        (data_layout == framework::DataLayout::kNCHW) + 1 + with_dynamic;
    int in_h = in_dim.d[h_axis];
    int in_w = in_dim.d[w_axis];

    auto* scale_var = scope.FindVar(op_desc.Input("Scale").front());
    if (scale_var != nullptr) {
      auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
      auto* scale_tensor = scale_var->GetMutable<framework::LoDTensor>();
      auto* scale_d = scale_tensor->data<float>();
      if (scale_tensor->numel() > 1) {
        scale_h = scale_d[0];
        scale_w = scale_d[1];
      } else {
        scale_h = scale_d[0];
        scale_w = scale_d[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of "
              "Operator(bilinear_interp_v2) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of "
              "Operator(bilinear_interp_v2) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      // get attr(scale)
      const std::vector<float> scale_attr =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("scale"));
      if (scale_attr.size() > 1) {
        scale_h = scale_attr[0];
        scale_w = scale_attr[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0, true,
            platform::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(bilinear_interp_v2) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0, true,
            platform::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(bilinear_interp_v2) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }

    if (scale_w > 0. && scale_h > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }

    if (!(scale_w > 0. && scale_h > 0.)) {
      scale_h = static_cast<float>(out_h) / static_cast<float>(in_h);
      scale_w = static_cast<float>(out_w) / static_cast<float>(in_w);
    }

    PADDLE_ENFORCE_EQ(
        scale_w > 0, true,
        platform::errors::InvalidArgument(
            "The scale_w in Attr(scale) of Operator(bilinear_interp_v2) "
            "should be greater than 0, but received value is %d.",
            scale_w));

    PADDLE_ENFORCE_EQ(
        scale_h > 0, true,
        platform::errors::InvalidArgument(
            "The scale_h in Attr(scale) of Operator(bilinear_interp_v2) "
            "should be greater than 0, but received value is %d.",
            scale_h));

    std::vector<float> scale;
    scale.reserve(3);

    if (engine_->with_dynamic_shape()) {
      scale.push_back(1.f);
    }

    if (data_layout == framework::DataLayout::kNCHW) {
      scale.push_back(1.f);
      scale.push_back(scale_h);
      scale.push_back(scale_w);
    } else if (data_layout == framework::DataLayout::kNHWC) {
      scale.push_back(scale_h);
      scale.push_back(scale_w);
      scale.push_back(1.f);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Data layout must be NCHW or NHWC."));
    }

    layer->setScales(scale.data(), scale.size());
    RreplenishLayerAndOutput(layer, "bilinear_interp_v2", {output_name},
                             test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(bilinear_interp_v2, BilinearInterpolateV2OpConverter);

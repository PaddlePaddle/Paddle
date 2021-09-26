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

class BilinearInterpolateOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid " << op_type_ << " op";

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);

    auto data_layout = framework::StringToDataLayout(
        BOOST_GET_CONST(std::string, op_desc.GetAttr("data_layout")));
    auto interp_method =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("interp_method"));

    auto resize_inputs = op_desc.Inputs();
    auto input_names = op_desc.Input("X");
    auto out_h = BOOST_GET_CONST(int, op_desc.GetAttr("out_h"));
    auto out_w = BOOST_GET_CONST(int, op_desc.GetAttr("out_w"));
    bool align_corners =
        BOOST_GET_CONST(bool, op_desc.GetAttr("align_corners"));
    auto align_mode = BOOST_GET_CONST(int, op_desc.GetAttr("align_mode"));

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    layer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);

#if IS_TRT_VERSION_GE(8016)
    if (align_mode == 0 && !align_corners) {
      layer->setCoordinateTransformation(
          nvinfer1::ResizeCoordinateTransformation::kHALF_PIXEL);
    } else {
      layer->setCoordinateTransformation(
          nvinfer1::ResizeCoordinateTransformation::kASYMMETRIC);
    }
#else
    PADDLE_THROW(platform::errors::Fatal(
        "bilinear_interp TRT converter is only supported on TRT "
        "8.0.1 or higher version."));
#endif

    auto in_dim = input->getDimensions();

    float scale_w = -1;
    float scale_h = -1;

    bool with_dynamic = engine_->with_dynamic_shape();
    int h_axis = (data_layout == framework::DataLayout::kNCHW) + with_dynamic;
    int w_axis =
        (data_layout == framework::DataLayout::kNCHW) + 1 + with_dynamic;
    int in_h = in_dim.d[h_axis];
    int in_w = in_dim.d[w_axis];

    bool has_scale_input = false;
    bool has_scale_input_attr =
        (resize_inputs.find("Scale") != resize_inputs.end());
    if (has_scale_input_attr) {
      has_scale_input = (op_desc.Input("Scale").size() > 0) ? true : false;
    }
    if (has_scale_input) {
      auto* scale_var = scope.FindVar(op_desc.Input("Scale")[0]);
      auto* scale_tensor = scale_var->GetMutable<framework::LoDTensor>();
      auto* scale_d = scale_tensor->data<float>();
      if (scale_tensor->numel() == 1 || op_type_ == "bilinear_interp") {  // v1
        scale_h = scale_d[0];
        scale_w = scale_d[0];
      } else if ((scale_tensor->numel() > 1) &&
                 (op_type_ != "bilinear_interp")) {
        scale_h = scale_d[0];
        scale_w = scale_d[1];
      } else {
        // no-op
      }

      PADDLE_ENFORCE_EQ(
          scale_w > 0, true,
          platform::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(%s) "
              "should be greater than 0, but received value is %d.",
              op_type_, scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0, true,
          platform::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(%s) "
              "should be greater than 0, but received value is %d.",
              op_type_, scale_h));
    } else {
      // get attr(scale)
      if (op_type_ == "bilinear_interp") {  // v1
        auto scale_val = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
        scale_h = scale_val;
        scale_w = scale_val;
      } else {  // v2
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
    }

    if (scale_w > 0. && scale_h > 0. && (!with_dynamic)) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }

    bool has_out_size_input = false;
    bool has_out_size_attr =
        (resize_inputs.find("OutSize") != resize_inputs.end());
    if (has_out_size_attr) {
      has_out_size_input = (op_desc.Input("OutSize").size() > 0) ? true : false;
    }
    if (has_out_size_input) {
      auto* out_size_var = scope.FindVar(op_desc.Input("OutSize")[0]);
      auto* out_size_tensor = out_size_var->GetMutable<framework::LoDTensor>();
      auto* out_size_d = out_size_tensor->data<int>();
      out_h = out_size_d[0];
      out_w = out_size_d[1];
      if ((out_h > 0 && out_w > 0) && (in_h > 0 && in_w > 0)) {
        scale_h = static_cast<float>(out_h) / static_cast<float>(in_h);
        scale_w = static_cast<float>(out_w) / static_cast<float>(in_w);
      }
    }

    if ((scale_h <= 0 || scale_w <= 0) && (!with_dynamic)) {
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
    RreplenishLayerAndOutput(layer, op_type_, {output_name}, test_mode);
  }

 protected:
  std::string op_type_ = "bilinear_interp";
};

class BilinearInterpolateV1OpConverter : public BilinearInterpolateOpConverter {
 public:
  BilinearInterpolateV1OpConverter() { op_type_ = "bilinear_interp"; }
};

class BilinearInterpolateV2OpConverter : public BilinearInterpolateOpConverter {
 public:
  BilinearInterpolateV2OpConverter() { op_type_ = "bilinear_interp_v2"; }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(bilinear_interp, BilinearInterpolateV1OpConverter);
REGISTER_TRT_OP_CONVERTER(bilinear_interp_v2, BilinearInterpolateV2OpConverter);

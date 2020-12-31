/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/operators/interpolate_op.h"

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

/*
 * ConcatOp
 */
class NearestInterpolateOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid nearest_interp op";
    //    std::cerr << "===NearestInterpolate Converter===" << std::endl;

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    /*
        std::cerr << "Dims: ";
        auto idim = input->getDimensions();
        for (int i = 0; i < idim.nbDims; ++ i) {
          std::cerr << idim.d[i] << " ";
        }
        std::cerr << std::endl;
    */
    auto data_layout =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("data_layout"));
    auto interp_method =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("interp_method"));
    bool align_corners =
        BOOST_GET_CONST(bool, op_desc.GetAttr("align_corners"));
    /*
        int align_mode = BOOST_GET_CONST(int, op_desc.GetAttr("align_mode"));

        int out_h = BOOST_GET_CONST(int, op_desc.GetAttr("out_h"));
        int out_w = BOOST_GET_CONST(int, op_desc.GetAttr("out_w"));
    */

    auto input_names = op_desc.Input("X");
    /*
        std::cerr << "input_names.size(): " << input_names.size() << std::endl;

        for (size_t i = 0; i < input_names.size(); ++ i) {
          std::cerr << i << ": " << input_names[i] << std::endl;
        }
    */
    auto scale = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
    //     std::cerr << "scale: " << scale << std::endl;
    std::vector<float> scales{1., scale, scale};

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    layer->setAlignCorners(align_corners);
    layer->setScales(scales.data(), scales.size());
    /*
        auto list_new_shape_tensor = BOOST_GET_CONST(framework::Tensor,
       op_desc.GetMultiInput("SizeTensor"));
        std::cerr << "list_new_shape_tensor.size(): " <<
       list_new_shape_tensor.size() << std::endl;
        if (list_new_shape_tensor.size() > 0) {
          // have size tensor
          auto new_size =
       paddle::operator::get_new_shape(list_new_shape_tensor);
          out_h = new_size[0];
          out_w = new_size[1];
        } else {
    */
    /*
          float scale;
          auto scale_tensor = BOOST_GET_CONST(framework::Tensor,
       op_desc.GetAttr("Scale"));
          if (scale_tensor != nullptr) {
            auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
            scale = scale_data[0];
          } else {
            scale = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
          }
          if (scale > 0) {
            out_h = static_cast<int>(in_h * scale);
            out_w = static_cast<int>(in_w * scale);
          }
          auto out_size = ctx.Input<Tensor>("OutSize");
          if (out_size != nullptr) {
            Tensor sizes;
            framework::TensorCopySync(*out_size, platform::CPUPlace(), &sizes);
            auto size_data = sizes.data<int>();
            out_h = size_data[0];
            out_w = size_data[1];
          }
        }
    */
    /*
        std::cerr << "data_layout: " << data_layout << std::endl;
        std::cerr << "interp_method: " << interp_method << std::endl;
        std::cerr << "align_corners: " << align_corners << std::endl;
        std::cerr << "align_mode: " << align_mode << std::endl;
        std::cerr << "out_h: " << out_h << std::endl;
        std::cerr << "out_w: " << out_w << std::endl;
    */
    RreplenishLayerAndOutput(layer, "nearest_interp", {output_name}, test_mode);

    //     std::cerr << "==================================" << std::endl;
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(nearest_interp, NearestInterpolateOpConverter);

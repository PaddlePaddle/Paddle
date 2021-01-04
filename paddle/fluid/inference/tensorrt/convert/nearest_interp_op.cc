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

    framework::OpDesc op_desc(op, nullptr);

    std::string input_name = op_desc.Input("X").front();
    std::string output_name = op_desc.Output("Out").front();

    auto input = engine_->GetITensor(input_name);
    auto data_layout =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("data_layout"));
    auto interp_method =
        BOOST_GET_CONST(std::string, op_desc.GetAttr("interp_method"));
    bool align_corners =
        BOOST_GET_CONST(bool, op_desc.GetAttr("align_corners"));

    auto input_names = op_desc.Input("X");
    auto scale = BOOST_GET_CONST(float, op_desc.GetAttr("scale"));
    std::vector<float> scales{1., scale, scale};

    auto layer = TRT_ENGINE_ADD_LAYER(engine_, Resize, *input);
    layer->setAlignCorners(align_corners);
    layer->setScales(scales.data(), scales.size());
    RreplenishLayerAndOutput(layer, "nearest_interp", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(nearest_interp, NearestInterpolateOpConverter);

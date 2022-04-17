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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/helper.h"

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
 * Stack converter from fluid to tensorRT.
 */
class RollOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid Roll op to tensorrt Slice layer";

    framework::OpDesc op_desc(op, nullptr);
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims input_dims = input->getDimensions();

    std::vector<int64_t> axis =
        BOOST_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("axis"));
    std::vector<int64_t> shifts =
        BOOST_GET_CONST(std::vector<int64_t>, op_desc.GetAttr("shifts"));

    nvinfer1::Dims start;
    start.nbDims = input_dims.nbDims;
    for (int i = 0; i < start.nbDims; i++) {
      start.d[i] = 0;
    }
    int axis_size = axis.size();
    for (int i = 0; i < axis_size; i++) {
      start.d[axis[i]] = (-shifts[i]) % input_dims.d[axis[i]];
    }

    nvinfer1::Dims stride;
    stride.nbDims = input_dims.nbDims;
    for (int i = 0; i < stride.nbDims; i++) {
      stride.d[i] = 1;
    }

    nvinfer1::Dims size;
    size.nbDims = input_dims.nbDims;
    for (int i = 0; i < size.nbDims; i++) {
      size.d[i] = 1;
    }

    auto output_name = op_desc.Output("Out")[0];

    auto shape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shape, *input);

    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, start, size, stride);
    layer->setInput(2, *shape_layer->getOutput(0));
#if IS_TRT_VERSION_GE(7000)
    layer->setMode(nvinfer1::SliceMode::kWRAP);
#endif

    RreplenishLayerAndOutput(layer, "roll", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(roll, RollOpConverter);

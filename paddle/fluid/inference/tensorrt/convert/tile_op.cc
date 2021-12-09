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
 * ReshapeOp
 */
class TileOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
#if IS_TRT_VERSION_GE(7000)
    VLOG(4) << "convert a fluid tile op to tensorrt tile layer";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);
    nvinfer1::Dims input_shape = input->getDimensions();
    std::vector<int> repeat_times =
        BOOST_GET_CONST(std::vector<int>, op_desc.GetAttr("repeat_times"));

    nvinfer1::Dims output_dim = input_shape;
    nvinfer1::Dims output_stride;
    // If input_dims.nbDims + 1 < repeat_times.size() means we
    // should expand 1 on batchsize. trt doesn't support this behavior.
    PADDLE_ENFORCE_GE(input_shape.nbDims + 1, repeat_times.size(),
                      platform::errors::InvalidArgument(
                          "Can't change batchsize, please check repeat_times"));
    int diff = input_shape.nbDims + 1 - repeat_times.size();
    if (diff > 0) repeat_times.insert(repeat_times.begin(), diff, 1);

    // Can't expand on batchsize
    PADDLE_ENFORCE_EQ(
        repeat_times[0], 1,
        platform::errors::InvalidArgument(
            "Can't expand on batchsize, please check repeat_times"));
    output_stride.nbDims = input_shape.nbDims;
    for (int i = 0; i < input_shape.nbDims; i++) {
      output_dim.d[i] = output_dim.d[i] * repeat_times[i + 1];
      output_stride.d[i] = 1;
    }

    auto* layer = TRT_ENGINE_ADD_LAYER(engine_, Slice, *input, input_shape,
                                       output_dim, output_stride);
    layer->setMode(nvinfer1::SliceMode::kWRAP);
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "tile", {output_name}, test_mode);
#endif
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(tile, TileOpConverter);

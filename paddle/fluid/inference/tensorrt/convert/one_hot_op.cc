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
 * OneHot Op
 */
class OneHotOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid one_hot op to tensorrt one_hot layer";
    framework::OpDesc op_desc(op, nullptr);

    const auto indices_tensor = engine_->GetITensor(op_desc.Input("X").front());
    const nvinfer1::ITensor* values_tensor;
    const nvinfer1::ITensor* depth_tensor;

    nvinfer1::Dims trt_values_tensor_shape;
    trt_values_tensor_shape.nbDims = 1;
    trt_values_tensor_shape.d[0] = 2;

    if (dtype == 2 || dtype == 3) {  // int, int64
      const std::vector<int> values_data = {0, 1};
      values_tensor = Add1DConstantLayer<int>(values_data, "values_tensor");
      if (dtype == 3) {  // int64
        VLOG(3) << "trt not support int64, so it is converted to int32.";
      }
    } else if (dtype == 5) {  // float
      const std::vector<float> values_data = {0.0f, 1.0f};
      values_tensor = Add1DConstantLayer<float>(values_data, "values_tensor");
    }

    nvinfer1::Dims indices_dims = indices_tensor->getDimensions();
    auto depth_name = op_desc.Input("depth_tensor");
    if (depth_name.size() == 0) {
      const int depth = PADDLE_GET_CONST(int, op_desc.GetAttr("depth"));
      int32_t length = 1;
      for (int32_t i = 0; i < indices_dims.nbDims; i++) {
        length *= indices_dims.d[i];
      }
      const std::vector<int> depth_data(length, depth);
      depth_tensor =
          Add1DConstantLayer<int>(depth_data, indices_dims, "values_tensor");
    } else {
      depth_tensor = engine_->GetITensor(depth_name.front());
    }
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, OneHot, *indices_tensor, *values_tensor, *depth_tensor, -1);

    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(layer, "one_hot", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(one_hot, OneHotOpConverter);

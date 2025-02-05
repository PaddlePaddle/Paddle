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

namespace paddle::framework {
class Scope;

}  // namespace paddle::framework
namespace paddle::framework::proto {
class OpDesc;
}  // namespace paddle::framework::proto

namespace paddle::inference::tensorrt {

/*
 * OneHot Op
 */
class OneHotOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
#if IS_TRT_VERSION_GE(8510)
    VLOG(3) << "convert a fluid one_hot op to tensorrt one_hot layer";
    framework::OpDesc op_desc(op, nullptr);

    const auto indices_tensor = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::ITensor* values_tensor{nullptr};
    nvinfer1::ITensor* depth_tensor{nullptr};
    const int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    if (dtype == 2 || dtype == 3) {  // int, int64
      const std::vector<int> values_data = {0, 1};
      values_tensor = Add1DConstantLayer<int>(values_data, "values_tensor");
      if (dtype == 3) {  // int64
        VLOG(3) << "trt not support int64, so it is converted to int32.";
      }
    } else if (dtype == 5 || dtype == 6) {  // float
      const std::vector<float> values_data = {0.0f, 1.0f};
      values_tensor = Add1DConstantLayer<float>(values_data, "values_tensor");
      if (dtype == 6) {  // int64
        VLOG(3) << "trt not support float64, so it is converted to float32.";
      }
    } else {
      PADDLE_THROW(common::errors::Fatal("one_hot is not supported"));
    }

    auto depth_name = op_desc.Input("depth_tensor");
    if (depth_name.empty()) {
      const int depth = PADDLE_GET_CONST(int, op_desc.GetAttr("depth"));
      depth_tensor = Add1DConstantLayer<int>(depth, "depth_tensor", true);
    } else {
      nvinfer1::Dims depth_dims;
      depth_dims.nbDims = 0;
      nvinfer1::ITensor* depth_tensor_paddle =
          engine_->GetITensor(depth_name.front());
      auto shuffle_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *depth_tensor_paddle);
      shuffle_layer->setReshapeDimensions(depth_dims);
      shuffle_layer->getOutput(0)->setName(depth_tensor_paddle->getName());
      depth_tensor = shuffle_layer->getOutput(0);
    }
    auto layer = TRT_ENGINE_ADD_LAYER(
        engine_, OneHot, *indices_tensor, *values_tensor, *depth_tensor, -1);

    auto output_name = op_desc.Output("Out").front();
    ReplenishLayerAndOutput(layer, "one_hot", {output_name}, test_mode);
#else
    VLOG(3) << "one_hot is not supported when TensorRT < 8.5.1";
#endif
  }
};

}  // namespace paddle::inference::tensorrt

REGISTER_TRT_OP_CONVERTER(one_hot, OneHotOpConverter);
REGISTER_TRT_OP_CONVERTER(one_hot_v2, OneHotOpConverter);

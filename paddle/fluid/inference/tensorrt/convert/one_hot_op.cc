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
    const int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));
    const bool allow_out_of_range =
        PADDLE_GET_CONST(int, op_desc.GetAttr("allow_out_of_range"));
    PADDLE_ENFORCE_EQ(allow_out_of_range,
                      false,
                      platform::errors::InvalidArgument(
                          "Errors occurs in Paddle-TRT one_hot op, "
                          "allow_out_of_range is not supported"));
    //    const int axis = ;

    nvinfer1::Dims trt_values_tensor_shape;
    trt_values_tensor_shape.nbDims = 1;
    trt_values_tensor_shape.d[0] = 2;

    if (dtype == 2) {  // int
      const int values_data[2] = {0, 1};
      values_tensor = AddConstantLayer<int>(
          values_data, trt_values_tensor_shape, "values_tensor");
    } else if (dtype == 3) {  // int64
      const int64_t values_data[2] = {0, 1};
      values_tensor = AddConstantLayer<int64_t>(
          values_data, trt_values_tensor_shape, "values_tensor");
    } else if (dtype == 5) {  // float
      const float values_data[2] = {0.0f, 1.0f};
      values_tensor = AddConstantLayer<float>(
          values_data, trt_values_tensor_shape, "values_tensor");
    }

    nvinfer1::Dims indices_dims = indices_tensor->getDimensions();
    auto depth_name = op_desc.Input("depth_tensor");
    if (depth_name.size() == 0) {
      const int depth = PADDLE_GET_CONST(int, op_desc.GetAttr("depth"));
      PADDLE_ENFORCE_GT(depth,
                        0,
                        platform::errors::InvalidArgument(
                            "Errors occurs in Paddle-TRT one_hot op, "
                            "axis must bigger than zero"));

      int32_t last_dim = 1;
      int32_t length = 1;
      for (int32_t i = 0; i < indices_dims.nbDims; i++) {
        last_dim = indices_dims.d[i];
        length *= last_dim;
      }
      if (last_dim == 1) {
        indices_dims.nbDims--;
      }
      const int* depth_data = new int[length]();
      depth_tensor =
          AddConstantLayer<float>(depth_data, indices_dims, "values_tensor");
    } else {
      depth_tensor = engine_->GetITensor(depth_name.front());
    }
    auto layer = TRT_ENGINE_ADD_LAYER(engine_,
                                      OneHot,
                                      *indices_tensor,
                                      *values_tensor,
                                      *depth_tensor,
                                      indices_dims.nbDims);

    auto output_name = op_desc.Output("Out").front();
    RreplenishLayerAndOutput(layer, "one_hot", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(one_hot, OneHotOpConverter);

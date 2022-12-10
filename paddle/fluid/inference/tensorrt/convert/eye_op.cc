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

#include <algorithm>
#include <iostream>
#include <iterator>

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
 * EyeOp.
 */
class EyeOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid eye op with tensorrt Constant layer";
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs attr
    const int num_rows = PADDLE_GET_CONST(int, op_desc.GetAttr("num_rows"));
    int num_columns = PADDLE_GET_CONST(int, op_desc.GetAttr("num_columns"));
    const int dtype = PADDLE_GET_CONST(int, op_desc.GetAttr("dtype"));

    // Set data dim
    nvinfer1::Dims input_shape;
    input_shape.nbDims = 2;
    if (-1 == num_columns) {
      num_columns = num_rows;
    }
    input_shape.d[0] = num_rows;
    input_shape.d[1] = num_columns;
    const int data_len = num_rows * num_columns;
    const int num_min = std::min(num_rows, num_columns);

    // Set data type
    void* trt_data = nullptr;
    nvinfer1::DataType nv_type = nvinfer1::DataType::kFLOAT;
    switch (dtype) {
      case 2:
        nv_type = nvinfer1::DataType::kINT32;
        std::unique_ptr<int32_t[]> data(new int32_t[data_len]());
        for (int i = 0; i < num_min; i++) {
          data[i * num_columns + i] = 1;
        }
        trt_data = static_cast<void*>(data.get());
        break;
      case 4:
        nv_type = nvinfer1::DataType::kHALF;
        std::unique_ptr<float16[]> data(new float16[data_len]());
        for (int i = 0; i < num_min; i++) {
          data[i * num_columns + i] = 1;
        }
        trt_data = static_cast<void*>(data.get());
        break;
      case 5:
        nv_type = nvinfer1::DataType::kFLOAT;
        std::unique_ptr<float[]> data(new float[data_len]());
        for (int i = 0; i < num_min; i++) {
          data[i * num_columns + i] = 1;
        }
        trt_data = static_cast<void*>(data.get());
        break;
      default:
        paddle::platform::errors::InvalidArgument(
            "Paddle-TRT loads weighths failed, found not supported data type "
            "%s.",
            dtype);
        break;
    }

    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_,
                             Constant,
                             input_shape,
                             nvinfer1::Weights{nv_type, trt_data, data_len});

    std::string output_name = op_desc.Output("Out").front();

    RreplenishLayerAndOutput(layer, "eye", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP_ITSELF(eye);
REGISTER_TRT_OP_CONVERTER(eye, EyeOpConverter);

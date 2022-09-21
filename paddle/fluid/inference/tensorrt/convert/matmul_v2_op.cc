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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/plugin/matmul_op_int8_plugin.h"

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
 * MatMulV2Op, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class MatMulV2OpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a fluid matmul_v2 op to tensorrt matmul layer ";
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);

    nvinfer1::Dims dims_x = input1->getDimensions();
    nvinfer1::Dims dims_y = input2->getDimensions();

    bool transpose_X = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_x"));
    bool transpose_Y = PADDLE_GET_CONST(bool, op_desc.GetAttr("trans_y"));

    auto output_name = op_desc.Output("Out")[0];

    nvinfer1::MatrixOperation matrix_operation_X =
        transpose_X ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_Y =
        transpose_Y ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;

    int one_num = 0;
    nvinfer1::ITensor* new_shape_tensor = nullptr;
    if (dims_x.nbDims < dims_y.nbDims) {
      one_num = dims_y.nbDims - dims_x.nbDims;
      new_shape_tensor = Shape(input1);
      std::vector<int32_t> one_vec(one_num, 1);
      auto* one_tensor = Add1DConstantLayer(one_vec);
      new_shape_tensor =
          Concat(std::vector<nvinfer1::ITensor*>{one_tensor, new_shape_tensor});

      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      reshape_layer->setInput(1, *new_shape_tensor);

      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   MatrixMultiply,
                                   *reshape_layer->getOutput(0),
                                   matrix_operation_X,
                                   *input2,
                                   matrix_operation_Y);

    } else if (dims_x.nbDims > dims_y.nbDims) {
      one_num = dims_x.nbDims - dims_y.nbDims;
      new_shape_tensor = Shape(input2);
      std::vector<int32_t> one_vec(one_num, 1);
      auto* one_tensor = Add1DConstantLayer(one_vec);
      new_shape_tensor =
          Concat(std::vector<nvinfer1::ITensor*>{one_tensor, new_shape_tensor});
      auto* reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input2);
      reshape_layer->setInput(1, *new_shape_tensor);

      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   MatrixMultiply,
                                   *input1,
                                   matrix_operation_X,
                                   *reshape_layer->getOutput(0),
                                   matrix_operation_Y);

    } else {
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   MatrixMultiply,
                                   *input1,
                                   matrix_operation_X,
                                   *input2,
                                   matrix_operation_Y);
    }
    VLOG(3) << "Convert a fluid matmul_v2_op_float to TensorRT ";

    RreplenishLayerAndOutput(layer, "matmul_v2_op", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul_v2, MatMulV2OpConverter);

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
 * After trt_map_ops_to_matrix_multiply_pass(mul, matmul, matmul_v2 ->
 * matrix_multiply), use MatrixMultiply layer, ElementWiseOperation::kPROD
 * layer.
 */
class MatrixMultiplyOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "convert a matrix_multiply op to tensorrt MatrixMultiply layer, "
               "ElementWiseOperation::kPROD layer.";
    framework::OpDesc op_desc(op, nullptr);

    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);
    auto output_name = op_desc.Output("Out")[0];

    nvinfer1::Dims dims_x = input1->getDimensions();
    int32_t x_rank = dims_x.nbDims;
    nvinfer1::Dims dims_y = input2->getDimensions();
    int32_t y_rank = dims_y.nbDims;

    int32_t x_num_col_dims =
        PADDLE_GET_CONST(int32_t, op_desc.GetAttr("x_num_col_dims"));
    if (x_num_col_dims < 0) {
      x_num_col_dims += x_rank;
    }

    // Temporarily solve the reformat problem of matrix multiplication, make
    // input.rank == 4. Possible solution in trt 8.7.
    if (x_rank == 2 && x_num_col_dims == 1 && engine_->use_varseqlen()) {
      VLOG(3) << "Temporarily solve the reformat problem of matrix "
                 "multiplication, make input.rank == 4. ";
      auto* reshape_before_matrix =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      std::vector<nvinfer1::ITensor*> reshape_before_tensor;
      reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 0));
      reshape_before_tensor.push_back(GetEleTensorOfShape(Shape(input1), 1));
      reshape_before_tensor.push_back(Add1DConstantLayer(1));
      reshape_before_tensor.push_back(Add1DConstantLayer(1));

      reshape_before_matrix->setInput(1, *Concat(reshape_before_tensor));
      reshape_before_matrix->setName(
          ("reshape_before_matrix(Output: " + output_name + ")").c_str());
      input1 = reshape_before_matrix->getOutput(0);
      dims_x = input1->getDimensions();
      x_rank = dims_x.nbDims;
    }

    if (x_num_col_dims != x_rank - 1) {
      std::vector<nvinfer1::ITensor*> before_shape_tensors;
      nvinfer1::ITensor* input_shape_tensor = Shape(input1);
      for (int i = 0; i < x_num_col_dims; ++i) {
        before_shape_tensors.push_back(
            GetEleTensorOfShape(input_shape_tensor, i));
      }
      nvinfer1::ITensor* producted = Add1DConstantLayer(1);
      for (int i = x_num_col_dims; i < x_rank; ++i) {
        producted = Prod(producted, GetEleTensorOfShape(input_shape_tensor, i));
      }
      before_shape_tensors.push_back(producted);
      nvinfer1::ITensor* before_shape_tensor = Concat(before_shape_tensors);
      auto* reshape_before_layer =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
      reshape_before_layer->setInput(1, *before_shape_tensor);
      reshape_before_layer->setName(
          ("reshape_x_before_matrix_multiply: Shuffle (Output: " + output_name +
           ")")
              .c_str());
      input1 = reshape_before_layer->getOutput(0);
      x_rank = x_num_col_dims + 1;
    }

    int32_t y_num_col_dims =
        PADDLE_GET_CONST(int32_t, op_desc.GetAttr("y_num_col_dims"));
    if (y_num_col_dims < 0) {
      y_num_col_dims += y_rank;
    }
    PADDLE_ENFORCE_EQ(
        y_num_col_dims,
        y_rank - 1,
        platform::errors::InvalidArgument(
            "The matrix_multiply op'y_num_col_dims should be equal "
            "to y'rank - 1, but got y_num_col_dims = %d, and y_rank = %d",
            y_num_col_dims,
            y_rank - 1));

    if (x_rank != 1 && y_rank != 1 && x_rank != y_rank) {
      if (x_rank < y_rank) {
        std::vector<nvinfer1::ITensor*> before_shape_tensors;
        nvinfer1::ITensor* input_shape_tensor = Shape(input1);
        for (int i = 0; i < y_rank - x_rank; ++i) {
          before_shape_tensors.push_back(Add1DConstantLayer(1));
        }
        for (int i = 0; i < x_rank; ++i) {
          before_shape_tensors.push_back(
              GetEleTensorOfShape(input_shape_tensor, i));
        }
        nvinfer1::ITensor* before_shape_tensor = Concat(before_shape_tensors);
        auto* reshape_before_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input1);
        reshape_before_layer->setInput(1, *before_shape_tensor);
        reshape_before_layer->setName(
            ("full_x_before_matrix_multiply: Shuffle (Output: " + output_name +
             ")")
                .c_str());
        input1 = reshape_before_layer->getOutput(0);
      } else {
        std::vector<nvinfer1::ITensor*> before_shape_tensors;
        nvinfer1::ITensor* input_shape_tensor = Shape(input2);

        for (int i = 0; i < x_rank - y_rank; ++i) {
          before_shape_tensors.push_back(Add1DConstantLayer(1));
        }
        for (int i = 0; i < y_rank; ++i) {
          before_shape_tensors.push_back(
              GetEleTensorOfShape(input_shape_tensor, i));
        }
        nvinfer1::ITensor* before_shape_tensor = Concat(before_shape_tensors);
        auto* reshape_before_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *input2);
        reshape_before_layer->setInput(1, *before_shape_tensor);
        reshape_before_layer->setName(
            ("full_y_before_matrix_multiply: Shuffle (Output: " + output_name +
             ")")
                .c_str());
        input2 = reshape_before_layer->getOutput(0);
      }
    }

    bool transpose_x = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_x"));
    bool transpose_y = PADDLE_GET_CONST(bool, op_desc.GetAttr("transpose_y"));

    nvinfer1::MatrixOperation matrix_operation_x =
        transpose_x ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_y =
        transpose_y ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;

    nvinfer1::ILayer* layer = nullptr;
    layer = TRT_ENGINE_ADD_LAYER(engine_,
                                 MatrixMultiply,
                                 *input1,
                                 matrix_operation_x,
                                 *input2,
                                 matrix_operation_y);

    engine_->SetTensorDynamicRange(input2, 1.0);

    float alpha = PADDLE_GET_CONST(float, op_desc.GetAttr("alpha"));
    if (alpha < 0.999 || alpha > 1.001) {
      auto* alpha_tensor = Add1DConstantLayer(alpha);
      std::vector<nvinfer1::ITensor*> alpha_shape_tensors;
      for (int i = 0; i < layer->getOutput(0)->getDimensions().nbDims; i++) {
        alpha_shape_tensors.push_back(Add1DConstantLayer(1));
      }
      auto* reshape_alpha =
          TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *alpha_tensor);
      reshape_alpha->setInput(1, *Concat(alpha_shape_tensors));
      layer = TRT_ENGINE_ADD_LAYER(engine_,
                                   ElementWise,
                                   *layer->getOutput(0),
                                   *reshape_alpha->getOutput(0),
                                   nvinfer1::ElementWiseOperation::kPROD);
    }
    RreplenishLayerAndOutput(
        layer, "matrix_multiply_op", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matrix_multiply, MatrixMultiplyOpConverter);

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
#include "paddle/fluid/inference/tensorrt/plugin/elementwise_op_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {

class ElementwiseTensorOpConverter : public OpConverter {
 public:
  ElementwiseTensorOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(3) << "Convert a fluid elementwise op to TensorRT IElementWiseLayer";
    framework::OpDesc op_desc(op, nullptr);
    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::ITensor* Y = nullptr;
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    if (Y_v) {
      // Y is weight
      auto* Y_t = Y_v->GetMutable<phi::DenseTensor>();
      std::vector<int> dims_y = phi::vectorize<int>(Y_t->dims());
      auto y_weight = engine_->GetTrtWeight(op_desc.Input("Y").front(), *Y_t);

      nvinfer1::Dims trt_dims_y;
      trt_dims_y.nbDims = dims_y.size();
      for (int i = 0; i < trt_dims_y.nbDims; i++) {
        trt_dims_y.d[i] = dims_y[i];
      }
      // this is the special case when dims_y includes batch dimension!
      // we need remove batch dimension!
      if (!engine_->with_dynamic_shape() &&
          trt_dims_y.nbDims == (X->getDimensions().nbDims + 1)) {
        trt_dims_y.nbDims--;
        PADDLE_ENFORCE_EQ(trt_dims_y.d[0],
                          1,
                          platform::errors::InvalidArgument(
                              "Elementwise type(%s) op's Y is a weight "
                              "including batch dimension. Please "
                              "check if the 0th dimension equals 1.",
                              op_type_));
        for (int i = 0; i < trt_dims_y.nbDims; i++) {
          trt_dims_y.d[i] = trt_dims_y.d[i + 1];
        }
      }
      Y = TRT_ENGINE_ADD_LAYER(engine_, Constant, trt_dims_y, y_weight.get())
              ->getOutput(0);
    } else {
      Y = engine_->GetITensor(op_desc.Input("Y").front());
    }
    bool swap_xy = false;
    // Swap X and Y
    if (X->getDimensions().nbDims < Y->getDimensions().nbDims) {
      auto* tmp = X;
      X = Y;
      Y = tmp;
      swap_xy = true;
    }
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();
    auto output_name = op_desc.Output("Out")[0];

    int axis = -1;
    // axis here is relative to explicit batch
    if (op_type_ != "logical_or" && op_type_ != "logical_xor" &&
        op_type_ != "logical_and") {
      axis = PADDLE_GET_CONST(int, op_desc.GetAttr("axis"));
    }
    int real_x_rank = dims_x.nbDims;
    int real_y_rank = dims_y.nbDims;
    if (!engine_->with_dynamic_shape()) {
      real_x_rank++;
      real_y_rank++;
      if (Y_v) real_y_rank--;
    }
    if (axis == -1) {
      axis = real_x_rank - real_y_rank;
    }
    if (!engine_->with_dynamic_shape() && axis > 0) {
      axis--;
    }

    // X: - -  -    - - - -
    //        axis
    // Y:      -    - -
    // we need expand Y's rank = X's rank
    int left_one_num = axis;
    int right_one_num = dims_x.nbDims - axis - dims_y.nbDims;
    nvinfer1::IShuffleLayer* reshape_layer;
    nvinfer1::ITensor* reshape_y_tensor;
    if (left_one_num > 0 || right_one_num > 0) {
      if (engine_->with_dynamic_shape()) {
        auto* y_shape_tensor = Shape(Y);
        auto* new_y_shape_tensor = y_shape_tensor;
        if (axis > 0) {
          std::vector<int32_t> left_one(left_one_num, 1);
          auto* left_one_tensor = Add1DConstantLayer(left_one);
          new_y_shape_tensor = Concat(std::vector<nvinfer1::ITensor*>{
              left_one_tensor, new_y_shape_tensor});
        }
        if (right_one_num > 0) {
          std::vector<int32_t> right_one(right_one_num, 1);
          auto* right_one_tensor = Add1DConstantLayer(right_one);
          new_y_shape_tensor = Concat(std::vector<nvinfer1::ITensor*>{
              new_y_shape_tensor, right_one_tensor});
        }
        reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *Y);
        reshape_layer->setInput(1, *new_y_shape_tensor);
      } else {
        nvinfer1::Dims new_y_dims;
        new_y_dims.nbDims = left_one_num + dims_y.nbDims + right_one_num;
        for (int i = 0; i < new_y_dims.nbDims; i++) new_y_dims.d[i] = 1;
        for (int i = 0; i < dims_y.nbDims; i++)
          new_y_dims.d[left_one_num + i] = dims_y.d[i];
        reshape_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *Y);
        reshape_layer->setReshapeDimensions(new_y_dims);
      }
      reshape_y_tensor = reshape_layer->getOutput(0);
    } else {
      // In fact , we can remove this `else`, but -> rt_resnet50_test CI in trt
      // 6015 faling, how ridiculous！
      reshape_y_tensor = Y;
    }

    // We should swap X and Y back, because some operators do not have symmetry
    if (swap_xy) {
      auto* tmp = reshape_y_tensor;
      reshape_y_tensor = X;
      X = tmp;
    }

    if (op_type_ == "less_equal") {
      auto* less_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *X,
                               *reshape_y_tensor,
                               nvinfer1::ElementWiseOperation::kLESS);
      auto* equal_layer =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *X,
                               *reshape_y_tensor,
                               nvinfer1::ElementWiseOperation::kEQUAL);
      auto* layer = TRT_ENGINE_ADD_LAYER(engine_,
                                         ElementWise,
                                         *(less_layer->getOutput(0)),
                                         *(equal_layer->getOutput(0)),
                                         nvinfer1::ElementWiseOperation::kOR);

      RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);
    } else {
      auto op_pair = ops.find(op_type_);
      PADDLE_ENFORCE_NE(
          op_pair,
          ops.end(),
          platform::errors::InvalidArgument(
              "Elementwise op's type(%s) is not supported. Please "
              "check if the op_type is correct.",
              op_type_));

      auto* layer = TRT_ENGINE_ADD_LAYER(
          engine_, ElementWise, *X, *reshape_y_tensor, op_pair->second);
      RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);
    }
  }

 protected:
  static const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
      ops;
  std::string op_type_;
};

const std::unordered_map<std::string, nvinfer1::ElementWiseOperation>
    ElementwiseTensorOpConverter::ops = {
        {"add", nvinfer1::ElementWiseOperation::kSUM},
        {"mul", nvinfer1::ElementWiseOperation::kPROD},
        {"sub", nvinfer1::ElementWiseOperation::kSUB},
        {"div", nvinfer1::ElementWiseOperation::kDIV},
        {"min", nvinfer1::ElementWiseOperation::kMIN},
        {"pow", nvinfer1::ElementWiseOperation::kPOW},
        {"max", nvinfer1::ElementWiseOperation::kMAX},
        {"floordiv", nvinfer1::ElementWiseOperation::kFLOOR_DIV},
        {"less_than", nvinfer1::ElementWiseOperation::kLESS},
        {"greater_than", nvinfer1::ElementWiseOperation::kGREATER},
        {"logical_or", nvinfer1::ElementWiseOperation::kOR},
        {"logical_xor", nvinfer1::ElementWiseOperation::kXOR},
        {"logical_and", nvinfer1::ElementWiseOperation::kAND},
};

class ElementwiseTensorAddOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorAddOpConverter() { op_type_ = "add"; }
};

class ElementwiseTensorMulOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMulOpConverter() { op_type_ = "mul"; }
};

class ElementwiseTensorSubOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorSubOpConverter() { op_type_ = "sub"; }
};

class ElementwiseTensorDivOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorDivOpConverter() { op_type_ = "div"; }
};

class ElementwiseTensorMinOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMinOpConverter() { op_type_ = "min"; }
};

class ElementwiseTensorMaxOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorMaxOpConverter() { op_type_ = "max"; }
};

class ElementwiseTensorPowOpConverter : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorPowOpConverter() { op_type_ = "pow"; }
};
class ElementwiseTensorFloorDivOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorFloorDivOpConverter() { op_type_ = "floordiv"; }
};
class ElementwiseTensorLessThanOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLessThanOpConverter() { op_type_ = "less_than"; }
};
class ElementwiseTensorGreaterThanOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorGreaterThanOpConverter() { op_type_ = "greater_than"; }
};
class ElementwiseTensorLogicalOrOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLogicalOrOpConverter() { op_type_ = "logical_or"; }
};
class ElementwiseTensorLogicalXorOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLogicalXorOpConverter() { op_type_ = "logical_xor"; }
};
class ElementwiseTensorLogicalAndOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLogicalAndOpConverter() { op_type_ = "logical_and"; }
};
class ElementwiseTensorLessEqualOpConverter
    : public ElementwiseTensorOpConverter {
 public:
  ElementwiseTensorLessEqualOpConverter() { op_type_ = "less_equal"; }
};
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(elementwise_add_weight,
                          ElementwiseTensorAddOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_mul_weight,
                          ElementwiseTensorMulOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_sub_weight,
                          ElementwiseTensorSubOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_div_weight,
                          ElementwiseTensorDivOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_max_weight,
                          ElementwiseTensorMaxOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_min_weight,
                          ElementwiseTensorMinOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_pow_weight,
                          ElementwiseTensorPowOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_floordiv_weight,
                          ElementwiseTensorFloorDivOpConverter);

REGISTER_TRT_OP_CONVERTER(elementwise_add_tensor,
                          ElementwiseTensorAddOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_sub_tensor,
                          ElementwiseTensorSubOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_div_tensor,
                          ElementwiseTensorDivOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_mul_tensor,
                          ElementwiseTensorMulOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_max_tensor,
                          ElementwiseTensorMaxOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_min_tensor,
                          ElementwiseTensorMinOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_pow_tensor,
                          ElementwiseTensorPowOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_floordiv_tensor,
                          ElementwiseTensorFloorDivOpConverter);
REGISTER_TRT_OP_CONVERTER(less_than, ElementwiseTensorLessThanOpConverter);
REGISTER_TRT_OP_CONVERTER(greater_than,
                          ElementwiseTensorGreaterThanOpConverter);
REGISTER_TRT_OP_CONVERTER(logical_or, ElementwiseTensorLogicalOrOpConverter);
REGISTER_TRT_OP_CONVERTER(logical_xor, ElementwiseTensorLogicalXorOpConverter);
REGISTER_TRT_OP_CONVERTER(logical_and, ElementwiseTensorLogicalAndOpConverter);
REGISTER_TRT_OP_CONVERTER(less_equal, ElementwiseTensorLessEqualOpConverter);

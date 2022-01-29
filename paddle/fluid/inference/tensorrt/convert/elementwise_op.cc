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

static bool CheckDims(const nvinfer1::Dims& dims_x,
                      const nvinfer1::Dims& dims_y) {
  if (dims_x.nbDims != dims_y.nbDims) {
    return false;
  }
  for (int i = 0; i < dims_x.nbDims; i++) {
    if (dims_x.d[i] != dims_y.d[i]) {
      return false;
    }
  }
  return true;
}

class ElementwiseWeightOpConverter : public OpConverter {
 public:
  ElementwiseWeightOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    nvinfer1::ILayer* layer = nullptr;
    framework::OpDesc op_desc(op, nullptr);
    VLOG(3) << "Convert a fluid elementwise op to TensorRT IScaleLayer";

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v, platform::errors::NotFound("Variable %s not found in scope.",
                                        op_desc.Input("Y").front().c_str()));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    float* weight_data = nullptr;
    auto output_name = op_desc.Output("Out")[0];
    weight_data =
        engine_->GetWeightCPUData(op_desc.Input("Y").front(), Y_t, false);
    nvinfer1::Dims dims_x = X->getDimensions();

    auto regist_eltwise_weight = [&](nvinfer1::ScaleMode scale_mode) {
      TensorRTEngine::Weight shift_weights{nvinfer1::DataType::kFLOAT,
                                           static_cast<void*>(weight_data),
                                           static_cast<size_t>(Y_t->numel())};
      TensorRTEngine::Weight scale_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};
      TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                           0};

      nvinfer1::IShuffleLayer* expand_layer = nullptr;
      nvinfer1::IShuffleLayer* squeeze_layer = nullptr;
      int dynamic_shape_offset = engine_->with_dynamic_shape() ? 1 : 0;
      auto input_dim = X->getDimensions();
      if (input_dim.nbDims < 3 + dynamic_shape_offset) {
        nvinfer1::Dims expand_shape;
        expand_shape.nbDims = 3 + dynamic_shape_offset;
        for (int i = 0; i < expand_shape.nbDims; i++) {
          if (i < input_dim.nbDims) {
            expand_shape.d[i] = input_dim.d[i] < 0 ? 0 : input_dim.d[i];
          } else {
            expand_shape.d[i] = 1;
          }
        }
        expand_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
        expand_layer->setReshapeDimensions(expand_shape);
        X = expand_layer->getOutput(0);
        expand_layer->getOutput(0)->setName(
            ("elementwise_reshape_out: " + output_name).c_str());
        expand_layer->setName(
            ("Elewise: Shuffle: (Output: " + output_name + ")").c_str());
      }
      if (op_type_ == "add") {
        nvinfer1::IScaleLayer* scale_layer = TRT_ENGINE_ADD_LAYER(
            engine_, ScaleNd, *X, scale_mode, shift_weights.get(),
            scale_weights.get(), power_weights.get(), dynamic_shape_offset);
        layer = scale_layer;
      } else if (op_type_ == "mul") {
        nvinfer1::IScaleLayer* scale_layer = TRT_ENGINE_ADD_LAYER(
            engine_, Scale, *X, scale_mode, scale_weights.get(),
            shift_weights.get(), power_weights.get());
        layer = scale_layer;
      }
      if (input_dim.nbDims < 3 + dynamic_shape_offset) {
        nvinfer1::Dims squeeze_shape;
        squeeze_shape.nbDims = input_dim.nbDims;
        for (int i = 0; i < squeeze_shape.nbDims; i++) {
          squeeze_shape.d[i] = input_dim.d[i] < 0 ? 0 : input_dim.d[i];
        }
        squeeze_layer =
            TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *(layer->getOutput(0)));
        squeeze_layer->setReshapeDimensions(squeeze_shape);
        RreplenishLayerAndOutput(squeeze_layer, "elementwise_" + op_type_,
                                 {output_name}, test_mode);
      } else {
        RreplenishLayerAndOutput(layer, "elementwise_" + op_type_,
                                 {output_name}, test_mode);
      }
      if (op_desc.HasAttr("enable_int8")) {
#if IS_TRT_VERSION_GE(5000)
        CHECK(op_desc.HasAttr("X_scale"));
        float x_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X_scale"));
        engine_->SetTensorDynamicRange(X, x_scale);
#endif
      }
    };

    if (engine_->with_dynamic_shape()) {
      if (Y_t->dims().size() == 1) {
        auto scale_mode = nvinfer1::ScaleMode::kCHANNEL;
        PADDLE_ENFORCE_EQ(Y_t->dims()[0], dims_x.d[1],
                          platform::errors::InvalidArgument(
                              "The Bias's size(%d) should be equal to the "
                              "first dim(%d) of the Input.",
                              Y_t->dims()[0], dims_x.d[1]));
        regist_eltwise_weight(scale_mode);
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The size of input bias's dims is %d, but TensorRT dynamic shape "
            "only support size = 1 for Elementwise op!",
            Y_t->dims().size()));
      }
      return;
    }

    std::vector<int> no_batch_dims;
    int start_index = 0;

    for (; start_index < dims_x.nbDims; start_index++)
      no_batch_dims.push_back(dims_x.d[start_index]);

    auto scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;

    std::vector<int> dims_y = framework::vectorize<int>(Y_t->dims());
    if (dims_y.size() == no_batch_dims.size() + 1) {
      if (dims_y[0] == 1) dims_y.erase(dims_y.begin());
    }

    if (dims_y.size() == 1 && dims_y[0] == no_batch_dims[0]) {
      scale_mode = nvinfer1::ScaleMode::kCHANNEL;
    } else if (dims_y.size() == no_batch_dims.size() &&
               dims_y[0] == no_batch_dims[0]) {
      scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;
      for (size_t i = 1; i < no_batch_dims.size(); i++) {
        if (dims_y[i] != no_batch_dims[i]) {
          scale_mode = nvinfer1::ScaleMode::kCHANNEL;
          break;
        }
      }
      if (scale_mode == nvinfer1::ScaleMode::kCHANNEL) {
        for (size_t i = 1; i < no_batch_dims.size(); i++) {
          if (dims_y[i] != 1)
            PADDLE_THROW(platform::errors::InvalidArgument(
                "The bias's %d dim is %d, but TensorRT dynamic shape only "
                "support it equals to 1 for Elementwise op!",
                i, dims_y[i]));
        }
      }
    } else {
      if (dims_y.size() >= 1) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The size of bias's dims is %d and bias's size is %d. TensorRT "
            "doesn't support this shape for Elementwise op!",
            dims_y.size(), dims_y[0]));
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The size of bias's dims is %d. TensorRT doesn't support "
            "this shape for Elementwise op!",
            dims_y.size()));
      }
    }
    regist_eltwise_weight(scale_mode);
  }

 protected:
  std::string op_type_;
};

class ElementwiseTensorOpConverter : public OpConverter {
 public:
  ElementwiseTensorOpConverter() {}
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    auto op_pair = ops.find(op_type_);
    PADDLE_ENFORCE_NE(op_pair, ops.end(),
                      platform::errors::InvalidArgument(
                          "Elementwise op's type(%s) is not supported. Please "
                          "check if the op_type is correct.",
                          op_type_));

    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);
    nvinfer1::ILayer* layer = nullptr;

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y = engine_->GetITensor(op_desc.Input("Y").front());
    std::vector<nvinfer1::ITensor*> itensors;
    itensors.push_back(X);
    itensors.push_back(Y);
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();

    int axis = BOOST_GET_CONST(int, op_desc.GetAttr("axis"));
    auto output_name = op_desc.Output("Out")[0];

    auto common_func = [&](nvinfer1::ILayer* layer) {
      RreplenishLayerAndOutput(layer, "elementwise", {output_name}, test_mode);
      if (op_desc.HasAttr("enable_int8")) {
#if IS_TRT_VERSION_GE(5000)
        CHECK(op_desc.HasAttr("X_scale"));
        CHECK(op_desc.HasAttr("Y_scale"));
        float x_scale = BOOST_GET_CONST(float, op_desc.GetAttr("X_scale"));
        float y_scale = BOOST_GET_CONST(float, op_desc.GetAttr("Y_scale"));
        engine_->SetTensorDynamicRange(X, x_scale);
        engine_->SetTensorDynamicRange(Y, y_scale);
#endif
      }
    };

    if (dims_x.nbDims == dims_y.nbDims) {
      // The two input tensor should have the same dims
      VLOG(3) << "Convert a fluid elementwise op to TensorRT IElementWiseLayer";
      nvinfer1::IElementWiseLayer* elet_layer =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *X, *Y, op_pair->second);

      layer = elet_layer;
    } else {
      VLOG(3) << "Convert a fluid elementwise op to TensorRT "
                 "ElementWisePluginLayer";
      if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        plugin::ElementwisePluginDynamic* plugin =
            new plugin::ElementwisePluginDynamic(op_type_, axis);
        layer = engine_->AddDynamicPlugin(itensors.data(), 2, plugin);
#else
        PADDLE_THROW(platform::errors::Fatal(
            "You are running the TRT Dynamic Shape mode, need to confirm that "
            "your TRT version is no less than 6.0"));
#endif
      } else {
        plugin::ElementWisePlugin* plugin =
            new plugin::ElementWisePlugin(op_type_, dims_x, dims_y, axis);

        std::vector<nvinfer1::ITensor*> inputs{X, Y};
        auto* plugin_layer = engine_->AddPlugin(
            inputs.data(), inputs.size(),
            reinterpret_cast<plugin::PluginTensorRT*>(plugin));

        layer = plugin_layer;
      }
    }
    common_func(layer);
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
};

class ElementwiseWeightAddOpConverter : public ElementwiseWeightOpConverter {
 public:
  ElementwiseWeightAddOpConverter() { op_type_ = "add"; }
};

class ElementwiseWeightMulOpConverter : public ElementwiseWeightOpConverter {
 public:
  ElementwiseWeightMulOpConverter() { op_type_ = "mul"; }
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

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(elementwise_add_weight,
                          ElementwiseWeightAddOpConverter);
REGISTER_TRT_OP_CONVERTER(elementwise_mul_weight,
                          ElementwiseWeightMulOpConverter);

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

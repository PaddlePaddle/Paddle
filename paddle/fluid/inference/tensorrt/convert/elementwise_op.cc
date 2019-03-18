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

    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);  // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    nvinfer1::Dims dims_x = X->getDimensions();
    PADDLE_ENFORCE(dims_x.nbDims >= 3, "x dims experts 3, but %d is given.",
                   dims_x.nbDims);

    auto* Y_v = scope.FindVar(op_desc.Input("Y").front());
    PADDLE_ENFORCE_NOT_NULL(Y_v);
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();

    platform::CPUPlace cpu_place;
    std::unique_ptr<framework::LoDTensor> weight_tensor(
        new framework::LoDTensor());
    weight_tensor->Resize(Y_t->dims());
    TensorCopySync((*Y_t), cpu_place, weight_tensor.get());
    auto* weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    auto scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;

    std::vector<int> dims_y = framework::vectorize2int(weight_tensor->dims());
    if (static_cast<int>(dims_y.size()) == dims_x.nbDims + 1) {
      if (dims_y[0] == 1) dims_y.erase(dims_y.begin());
    }

    if (static_cast<int>(dims_y.size()) == 1 && dims_y[0] == dims_x.d[0]) {
      scale_mode = nvinfer1::ScaleMode::kCHANNEL;
    } else if (static_cast<int>(dims_y.size()) == dims_x.nbDims &&
               dims_y[0] == dims_x.d[0]) {
      scale_mode = nvinfer1::ScaleMode::kELEMENTWISE;
      for (int i = 1; i < dims_x.nbDims; i++) {
        if (dims_y[i] != dims_x.d[i]) {
          scale_mode = nvinfer1::ScaleMode::kCHANNEL;
          break;
        }
      }
      if (scale_mode == nvinfer1::ScaleMode::kCHANNEL) {
        for (int i = 1; i < dims_x.nbDims; i++) {
          if (dims_y[i] != 1)
            PADDLE_THROW(
                "TensorRT unsupported weight shape for Elementwise op!");
        }
      }
    } else {
      PADDLE_THROW("TensorRT unsupported weight Shape for Elementwise op!");
    }

    TensorRTEngine::Weight shift_weights{
        nvinfer1::DataType::kFLOAT, static_cast<void*>(weight_data),
        weight_tensor->memory_size() / sizeof(float)};
    TensorRTEngine::Weight scale_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    TensorRTEngine::Weight power_weights{nvinfer1::DataType::kFLOAT, nullptr,
                                         0};
    if (op_type_ == "add") {
      nvinfer1::IScaleLayer* scale_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *X, scale_mode, shift_weights.get(),
          scale_weights.get(), power_weights.get());
      layer = scale_layer;
    } else if (op_type_ == "mul") {
      nvinfer1::IScaleLayer* scale_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *X, scale_mode, scale_weights.get(),
          shift_weights.get(), power_weights.get());
      layer = scale_layer;
    }

    auto output_name = op_desc.Output("Out")[0];
    layer->setName(
        ("elementwise_" + op_type_ + "(Output: " + output_name + ")").c_str());
    layer->getOutput(0)->setName(output_name.c_str());
    engine_->weight_map[op_desc.Input("Y").front()] = std::move(weight_tensor);
    engine_->SetITensor(output_name, layer->getOutput(0));
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
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
    PADDLE_ENFORCE(op_pair != ops.end(), "Wrong elementwise op type!");

    // Here the two nullptr looks strange, that's because the
    // framework::OpDesc's constructor is strange.
    framework::OpDesc op_desc(op, nullptr);

    PADDLE_ENFORCE_EQ(op_desc.Input("X").size(), 1);
    PADDLE_ENFORCE_EQ(op_desc.Input("Y").size(), 1);  // Y is a weight
    PADDLE_ENFORCE_EQ(op_desc.Output("Out").size(), 1);

    auto* X = engine_->GetITensor(op_desc.Input("X").front());
    auto* Y = engine_->GetITensor(op_desc.Input("Y").front());
    nvinfer1::Dims dims_x = X->getDimensions();
    nvinfer1::Dims dims_y = Y->getDimensions();

    int axis = boost::get<int>(op_desc.GetAttr("axis"));
    auto output_name = op_desc.Output("Out")[0];
    if (CheckDims(dims_x, dims_y)) {
      // The two input tensor should have the same dims
      VLOG(3) << "Convert a fluid elementwise op to TensorRT IElementWiseLayer";
      nvinfer1::IElementWiseLayer* layer = TRT_ENGINE_ADD_LAYER(
          engine_, ElementWise, *const_cast<nvinfer1::ITensor*>(X),
          *const_cast<nvinfer1::ITensor*>(Y), op_pair->second);

      layer->setName(("elementwise (Output: " + output_name + ")").c_str());
      layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(0));
    } else {
      VLOG(3) << "Convert a fluid elementwise op to TensorRT "
                 "ElementWisePluginLayer";

      plugin::ElementWisePlugin* plugin =
          new plugin::ElementWisePlugin(op_type_, dims_x, dims_y, axis);
      plugin->AddInput(X);
      plugin->AddInput(Y);
      nvinfer1::IPluginLayer* layer = engine_->AddPlugin(
          const_cast<nvinfer1::ITensor* const*>(plugin->GetInputs().data()), 2,
          reinterpret_cast<plugin::PluginTensorRT*>(plugin));

      layer->setName(("elementwise (Output: " + output_name + ")").c_str());
      layer->getOutput(0)->setName(output_name.c_str());
      engine_->SetITensor(output_name, layer->getOutput(0));
    }
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
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

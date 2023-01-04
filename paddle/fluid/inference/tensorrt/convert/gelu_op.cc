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
#include "paddle/fluid/inference/tensorrt/plugin/gelu_op_plugin.h"

namespace nvinfer1 {
class ILayer;
}  // namespace nvinfer1
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
 * Gelu converter from fluid to tensorRT.
 */
/*
 * Gelu converter from fluid to tensorRT.
 */
class GeluOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope,
                  bool test_mode) override {
    VLOG(4) << "convert fluid gelu op to tensorrt gelu layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    nvinfer1::ILayer* layer = nullptr;
    if (op_desc.HasAttr("approximate") &&
        PADDLE_GET_CONST(bool, op_desc.GetAttr("approximate"))) {
#if IS_TRT_VERSION_GE(7000)
      nvinfer1::Dims input_shape;
      input_shape.nbDims = input->getDimensions().nbDims;
      for (int i = 0; i < input_shape.nbDims; ++i) {
        input_shape.d[i] = 1;
      }
      std::string out_name = op_desc.Output("Out").front();
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_gelu_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };
      float* constant_pow = create_weights(3.0f, "constant_pow");
      float* constant_multiply = create_weights(0.044715f, "constant_multiply");
      float* constant_sqrt =
          create_weights(0.79788456080286535587989211986876f, "constant_sqrt");
      float* constant_one = create_weights(1.0f, "constant_one");
      float* constant_half = create_weights(0.5f, "constant_half");
      auto constant_layer_pow = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{
              nvinfer1::DataType::kFLOAT, static_cast<void*>(constant_pow), 1});
      auto constant_layer_multiply = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(constant_multiply),
                            1});
      auto constant_layer_sqrt = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(constant_sqrt),
                            1});
      auto constant_layer_one = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{
              nvinfer1::DataType::kFLOAT, static_cast<void*>(constant_one), 1});
      auto constant_layer_half = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(constant_half),
                            1});
      auto layer_pow =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *input,
                               *constant_layer_pow->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPOW);
      auto layer_mul =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_pow->getOutput(0),
                               *constant_layer_multiply->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPROD);
      auto layer_add =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_mul->getOutput(0),
                               *input,
                               nvinfer1::ElementWiseOperation::kSUM);
      auto layer_sqrt =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_add->getOutput(0),
                               *constant_layer_sqrt->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPROD);
      auto layer_tanh = TRT_ENGINE_ADD_LAYER(engine_,
                                             Activation,
                                             *layer_sqrt->getOutput(0),
                                             nvinfer1::ActivationType::kTANH);
      auto layer_one =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_tanh->getOutput(0),
                               *constant_layer_one->getOutput(0),
                               nvinfer1::ElementWiseOperation::kSUM);
      auto layer_CDF =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_one->getOutput(0),
                               *constant_layer_half->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPROD);
      auto y = TRT_ENGINE_ADD_LAYER(engine_,
                                    ElementWise,
                                    *layer_CDF->getOutput(0),
                                    *input,
                                    nvinfer1::ElementWiseOperation::kPROD);
      layer = y;
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running GeLU Op with approximate True, need to confirm that "
          "your TRT version is no less than 7.0"));
#endif
    } else {
#if IS_TRT_VERSION_GE(7000)
      nvinfer1::Dims input_shape;
      input_shape.nbDims = input->getDimensions().nbDims;
      for (int i = 0; i < input_shape.nbDims; ++i) {
        input_shape.d[i] = 1;
      }
      std::string out_name = op_desc.Output("Out").front();
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<phi::DenseTensor> tmp_tensor(new phi::DenseTensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_gelu_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };
      float* constant_one = create_weights(1.0f, "constant_one");
      float* constant_half = create_weights(0.5f, "constant_half");
      float* constant_rsqrt2 =
          create_weights(0.70710678118f, "constant_rsqrt2");
      auto constant_layer_one = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{
              nvinfer1::DataType::kFLOAT, static_cast<void*>(constant_one), 1});
      auto constant_layer_half = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(constant_half),
                            1});
      auto constant_layer_rsqrt2 = TRT_ENGINE_ADD_LAYER(
          engine_,
          Constant,
          input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(constant_rsqrt2),
                            1});
      auto layer_mul =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *input,
                               *constant_layer_rsqrt2->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPROD);
      auto layer_erf = TRT_ENGINE_ADD_LAYER(engine_,
                                            Unary,
                                            *layer_mul->getOutput(0),
                                            nvinfer1::UnaryOperation::kERF);
      auto layer_add =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_erf->getOutput(0),
                               *constant_layer_one->getOutput(0),
                               nvinfer1::ElementWiseOperation::kSUM);
      auto layer_CDF =
          TRT_ENGINE_ADD_LAYER(engine_,
                               ElementWise,
                               *layer_add->getOutput(0),
                               *constant_layer_half->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPROD);
      auto y = TRT_ENGINE_ADD_LAYER(engine_,
                                    ElementWise,
                                    *layer_CDF->getOutput(0),
                                    *input,
                                    nvinfer1::ElementWiseOperation::kPROD);
      layer = y;
#else  // if IS_TRT_VERSION_GE(7000)
      int input_num = op_desc.Input("X").size();
      if (engine_->with_dynamic_shape()) {
#if IS_TRT_VERSION_GE(6000)
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        plugin::GeluPluginDynamic* plugin =
            new plugin::GeluPluginDynamic(with_fp16);
        layer = engine_->AddDynamicPlugin(&input, input_num, plugin);
#else
        PADDLE_THROW(platform::errors::Fatal(
            "You are running the TRT Dynamic Shape mode, need to confirm that "
            "your TRT version is no less than 6.0"));
#endif
      } else {
        bool with_fp16 =
            engine_->WithFp16() && !engine_->disable_trt_plugin_fp16();
        plugin::GeluPlugin* plugin = new plugin::GeluPlugin(with_fp16);
        layer = engine_->AddPlugin(&input, input_num, plugin);
      }
#endif  // if IS_TRT_VERSION_GE(7000)
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "gelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(gelu, GeluOpConverter);

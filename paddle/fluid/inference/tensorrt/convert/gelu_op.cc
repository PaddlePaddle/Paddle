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
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(4) << "convert fluid gelu op to tensorrt gelu layer";
    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input = engine_->GetITensor(op_desc.Input("X")[0]);

    nvinfer1::ILayer* layer = nullptr;
    if (BOOST_GET_CONST(bool, op_desc.GetAttr("approximate"))) {
#if IS_TRT_VERSION_GE(7000)
      nvinfer1::Dims input_shape;
      input_shape.nbDims = input->getDimensions().nbDims;
      for (int i = 0; i < input_shape.nbDims; ++i) {
        input_shape.d[i] = 1;
      }
      std::string out_name = op_desc.Output("Out").front();
      auto create_weights = [&](float data, std::string type) -> float* {
        std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(out_name + "_gelu_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };
      float* fPow = create_weights(3.0f, "fPow");
      float* fMultiply = create_weights(0.044715f, "fMultiply");
      float* fSqrt =
          create_weights(0.79788456080286535587989211986876f, "fSqrt");
      float* fOne = create_weights(1.0f, "fOne");
      float* fHalf = create_weights(0.5f, "fHalf");
      /*
      static const float fPow = 3.0f;
      static const float fMultiply = 0.044715f;
      static const float fSqrt = 0.79788456080286535587989211986876f;
      static const float fOne = 1.0f;
      static const float fHalf = 0.5f;
      */
      auto POW =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape,
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                 static_cast<void*>(fPow), 1});
      auto MULTIPLY = TRT_ENGINE_ADD_LAYER(
          engine_, Constant, input_shape,
          nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                            static_cast<void*>(fMultiply), 1});
      auto SQRT =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape,
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                 static_cast<void*>(fSqrt), 1});
      auto ONE =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape,
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                 static_cast<void*>(fOne), 1});
      auto HALF =
          TRT_ENGINE_ADD_LAYER(engine_, Constant, input_shape,
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT,
                                                 static_cast<void*>(fHalf), 1});
      auto X_pow =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *input, *POW->getOutput(0),
                               nvinfer1::ElementWiseOperation::kPOW);
      auto X_mul = TRT_ENGINE_ADD_LAYER(
          engine_, ElementWise, *X_pow->getOutput(0), *MULTIPLY->getOutput(0),
          nvinfer1::ElementWiseOperation::kPROD);
      auto X_add =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *X_mul->getOutput(0),
                               *input, nvinfer1::ElementWiseOperation::kSUM);
      auto X_sqrt = TRT_ENGINE_ADD_LAYER(
          engine_, ElementWise, *X_add->getOutput(0), *SQRT->getOutput(0),
          nvinfer1::ElementWiseOperation::kPROD);
      auto X_tanh =
          TRT_ENGINE_ADD_LAYER(engine_, Activation, *X_sqrt->getOutput(0),
                               nvinfer1::ActivationType::kTANH);
      auto X_one = TRT_ENGINE_ADD_LAYER(
          engine_, ElementWise, *X_tanh->getOutput(0), *ONE->getOutput(0),
          nvinfer1::ElementWiseOperation::kSUM);
      auto CDF = TRT_ENGINE_ADD_LAYER(engine_, ElementWise,
                                      *X_one->getOutput(0), *HALF->getOutput(0),
                                      nvinfer1::ElementWiseOperation::kPROD);
      auto y =
          TRT_ENGINE_ADD_LAYER(engine_, ElementWise, *CDF->getOutput(0), *input,
                               nvinfer1::ElementWiseOperation::kPROD);
      layer = y;
#else
      PADDLE_THROW(platform::errors::Fatal(
          "You are running GeLU Op with approximate True, need to confirm that "
          "your TRT version is no less than 7.0"));
#endif
    } else {
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
    }
    auto output_name = op_desc.Output("Out")[0];
    RreplenishLayerAndOutput(layer, "gelu", {output_name}, test_mode);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(gelu, GeluOpConverter);
